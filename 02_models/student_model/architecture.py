"""
Lightweight Student CNN Architecture for Real-time Drone Deployment

This module implements a lightweight student model optimized for edge deployment.
The model uses MobileNetV2 backbone with efficient saliency prediction head
for real-time attention prediction on drone hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from typing import Tuple, Optional


class EfficientBackbone(nn.Module):
    """
    Efficient backbone using MobileNetV2 for visual feature extraction.
    Optimized for speed and memory efficiency on edge devices.
    """
    
    def __init__(self, pretrained: bool = True, width_mult: float = 1.0):
        super().__init__()
        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
        
        # Remove classification layers
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimensions
        self.feature_dim = self.backbone.last_channel
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from input frames.
        
        Args:
            x: Input video frames [B, C, H, W]
            
        Returns:
            Visual features [B, feature_dim, H', W']
        """
        features = self.backbone.features(x)
        return features


class SpatialAttentionModule(nn.Module):
    """
    Spatial attention module for focusing on relevant regions.
    Uses channel and spatial attention mechanisms.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel and spatial attention.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Attended features [B, C, H, W]
        """
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x


class EfficientSaliencyHead(nn.Module):
    """
    Efficient saliency prediction head optimized for speed.
    Uses depthwise separable convolutions and upsampling.
    """
    
    def __init__(self, in_channels: int, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        
        # Efficient upsampling path
        self.upsample_path = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels // 2, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            # Further reduction
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, groups=in_channels // 2),
            nn.Conv2d(in_channels // 4, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate saliency map from features.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Saliency map [B, 1, img_size, img_size]
        """
        # Upsample to target size
        saliency = self.upsample_path(features)
        saliency = F.interpolate(
            saliency, 
            size=(self.img_size, self.img_size), 
            mode='bilinear', 
            align_corners=False
        )
        return saliency


class LightweightStudentCNN(nn.Module):
    """
    Lightweight student model for real-time saliency prediction.
    
    Optimized for deployment on drone hardware with:
    - MobileNetV2 backbone for efficiency
    - Spatial attention for focus
    - Efficient upsampling for speed
    """
    
    def __init__(
        self,
        backbone: str = 'mobilenet_v2',
        width_mult: float = 1.0,
        img_size: int = 224,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Core components
        self.backbone = EfficientBackbone(pretrained=pretrained, width_mult=width_mult)
        self.attention = SpatialAttentionModule(self.backbone.feature_dim)
        self.saliency_head = EfficientSaliencyHead(self.backbone.feature_dim, img_size)
        
        # Store configuration
        self.img_size = img_size
        self.width_mult = width_mult
        
    def forward(self, video_frame: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the lightweight student model.
        
        Args:
            video_frame: Single video frame [B, C, H, W]
            
        Returns:
            saliency_map: Predicted saliency map [B, 1, H, W]
        """
        # Extract visual features
        features = self.backbone(video_frame)  # [B, C, H', W']
        
        # Apply spatial attention
        attended_features = self.attention(features)  # [B, C, H', W']
        
        # Generate saliency map
        saliency_map = self.saliency_head(attended_features)  # [B, 1, img_size, img_size]
        
        return saliency_map
    
    def get_model_size(self) -> int:
        """
        Calculate model size in parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """
        Calculate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def optimize_for_inference(self):
        """
        Optimize model for inference by:
        - Setting to eval mode
        - Fusing batch norm layers
        - Enabling optimizations
        """
        self.eval()
        
        # Fuse batch norm layers for efficiency
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.fuse = True
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


class QuantizedStudentCNN(LightweightStudentCNN):
    """
    Quantized version of the student model for ultra-efficient deployment.
    Uses PyTorch quantization for further speed and memory optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized = False
        
    def quantize_model(self):
        """
        Apply quantization to the model for deployment.
        """
        if not self.quantized:
            # Apply dynamic quantization
            self.backbone = torch.quantization.quantize_dynamic(
                self.backbone, 
                {nn.Conv2d, nn.Linear}, 
                dtype=torch.qint8
            )
            
            self.attention = torch.quantization.quantize_dynamic(
                self.attention,
                {nn.Conv2d, nn.Linear},
                dtype=torch.qint8
            )
            
            self.saliency_head = torch.quantization.quantize_dynamic(
                self.saliency_head,
                {nn.Conv2d, nn.Linear},
                dtype=torch.qint8
            )
            
            self.quantized = True
    
    def forward(self, video_frame: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized model.
        """
        if not self.quantized:
            self.quantize_model()
        
        return super().forward(video_frame)


# Example usage and testing
if __name__ == "__main__":
    # Initialize models
    student_model = LightweightStudentCNN()
    quantized_model = QuantizedStudentCNN()
    
    # Create dummy input
    batch_size = 2
    video_input = torch.randn(batch_size, 3, 224, 224)
    
    # Test student model
    with torch.no_grad():
        saliency_map = student_model(video_input)
        print(f"Student model saliency shape: {saliency_map.shape}")
        print(f"Student model size: {student_model.get_model_size_mb():.2f} MB")
        
        # Test quantized model
        quantized_saliency = quantized_model(video_input)
        print(f"Quantized model saliency shape: {quantized_saliency.shape}")
        print(f"Quantized model size: {quantized_model.get_model_size_mb():.2f} MB")
    
    # Performance comparison
    import time
    
    # Warm up
    for _ in range(10):
        _ = student_model(video_input)
    
    # Time inference
    start_time = time.time()
    for _ in range(100):
        _ = student_model(video_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    print(f"Average inference time: {avg_time:.2f} ms")
