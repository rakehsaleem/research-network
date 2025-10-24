"""
Multimodal Teacher ViT Architecture for Expert Attention Prediction

This module implements the teacher model that combines visual features from video frames
with gaze history to predict saliency maps. The model uses a Vision Transformer backbone
with multimodal fusion capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Tuple, Dict, Optional


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for visual feature extraction.
    Uses pre-trained ViT model from timm library.
    """
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True):
        super().__init__()
        self.vit = create_model(model_name, pretrained=pretrained)
        # Remove classification head
        self.vit.head = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from input frames.
        
        Args:
            x: Input video frames [B, C, H, W]
            
        Returns:
            Visual features [B, num_patches, embed_dim]
        """
        # Extract patch embeddings and transformer features
        features = self.vit.forward_features(x)
        return features


class GazeTemporalEncoder(nn.Module):
    """
    LSTM/GRU encoder for processing gaze history.
    Captures temporal dependencies in eye-tracking data.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2  # Bidirectional
        
    def forward(self, gaze_history: torch.Tensor) -> torch.Tensor:
        """
        Process gaze history through LSTM.
        
        Args:
            gaze_history: Gaze coordinates [B, seq_len, 2]
            
        Returns:
            Encoded gaze features [B, hidden_dim * 2]
        """
        lstm_out, (hidden, cell) = self.lstm(gaze_history)
        # Use last hidden state
        return lstm_out[:, -1, :]


class MultimodalFusionBlock(nn.Module):
    """
    Fusion block for combining visual and gaze features.
    Uses attention mechanism to align modalities.
    """
    
    def __init__(self, visual_dim: int, gaze_dim: int, output_dim: int = 512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.gaze_proj = nn.Linear(gaze_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, visual_features: torch.Tensor, gaze_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and gaze features using attention.
        
        Args:
            visual_features: Visual features [B, num_patches, visual_dim]
            gaze_features: Gaze features [B, gaze_dim]
            
        Returns:
            Fused features [B, num_patches, output_dim]
        """
        # Project features to common dimension
        visual_proj = self.visual_proj(visual_features)
        gaze_proj = self.gaze_proj(gaze_features).unsqueeze(1)  # [B, 1, output_dim]
        
        # Use gaze as query, visual as key and value
        fused_features, _ = self.attention(
            query=gaze_proj.expand(-1, visual_proj.size(1), -1),
            key=visual_proj,
            value=visual_proj
        )
        
        # Residual connection and normalization
        output = self.norm(fused_features + visual_proj)
        return output


class SaliencyPredictionHead(nn.Module):
    """
    Prediction head for generating saliency maps.
    Converts fused features to spatial attention maps.
    """
    
    def __init__(self, input_dim: int, patch_size: int = 16, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Generate saliency map from fused features.
        
        Args:
            fused_features: Fused multimodal features [B, num_patches, input_dim]
            
        Returns:
            Saliency map [B, 1, H, W]
        """
        # Predict saliency for each patch
        patch_saliency = self.head(fused_features)  # [B, num_patches, 1]
        
        # Reshape to spatial map
        batch_size = patch_saliency.size(0)
        spatial_size = int(self.num_patches ** 0.5)
        saliency_map = patch_saliency.view(batch_size, spatial_size, spatial_size)
        
        # Upsample to original image size
        saliency_map = F.interpolate(
            saliency_map.unsqueeze(1),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        return saliency_map


class MultimodalTeacherViT(nn.Module):
    """
    Complete multimodal teacher model for attention prediction.
    
    Combines visual features from ViT with gaze history through
    temporal encoding and multimodal fusion.
    """
    
    def __init__(
        self,
        vit_model: str = 'vit_base_patch16_224',
        visual_dim: int = 768,
        gaze_dim: int = 512,
        fusion_dim: int = 512,
        patch_size: int = 16,
        img_size: int = 224
    ):
        super().__init__()
        
        # Core components
        self.visual_backbone = ViTBackbone(vit_model, pretrained=True)
        self.gaze_encoder = GazeTemporalEncoder(input_dim=2, hidden_dim=gaze_dim//2)
        self.fusion_block = MultimodalFusionBlock(visual_dim, gaze_dim, fusion_dim)
        self.saliency_head = SaliencyPredictionHead(fusion_dim, patch_size, img_size)
        
        # Store dimensions for feature extraction
        self.visual_dim = visual_dim
        self.gaze_dim = gaze_dim
        self.fusion_dim = fusion_dim
        
    def forward(
        self, 
        video_features: torch.Tensor, 
        gaze_history: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the multimodal teacher model.
        
        Args:
            video_features: Video frame features [B, C, H, W]
            gaze_history: Gaze coordinates [B, seq_len, 2]
            
        Returns:
            saliency_map: Predicted saliency map [B, 1, H, W]
            intermediate_features: Dictionary of intermediate features for distillation
        """
        # Extract visual features
        visual_features = self.visual_backbone(video_features)  # [B, num_patches, visual_dim]
        
        # Encode gaze history
        gaze_features = self.gaze_encoder(gaze_history)  # [B, gaze_dim]
        
        # Fuse multimodal features
        fused_features = self.fusion_block(visual_features, gaze_features)  # [B, num_patches, fusion_dim]
        
        # Generate saliency map
        saliency_map = self.saliency_head(fused_features)  # [B, 1, H, W]
        
        # Prepare intermediate features for knowledge distillation
        intermediate_features = {
            'visual_features': visual_features,
            'gaze_features': gaze_features,
            'fused_features': fused_features,
            'patch_saliency': fused_features  # Features before final prediction
        }
        
        return saliency_map, intermediate_features
    
    def extract_features(self, video_features: torch.Tensor, gaze_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            video_features: Video frame features [B, C, H, W]
            gaze_history: Gaze coordinates [B, seq_len, 2]
            
        Returns:
            Dictionary of intermediate features
        """
        with torch.no_grad():
            _, features = self.forward(video_features, gaze_history)
            return features


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = MultimodalTeacherViT()
    
    # Create dummy inputs
    batch_size = 2
    video_input = torch.randn(batch_size, 3, 224, 224)
    gaze_input = torch.randn(batch_size, 10, 2)  # 10 timesteps of gaze data
    
    # Forward pass
    saliency_map, features = model(video_input, gaze_input)
    
    print(f"Saliency map shape: {saliency_map.shape}")
    print(f"Available intermediate features: {list(features.keys())}")
    for key, value in features.items():
        print(f"{key}: {value.shape}")
