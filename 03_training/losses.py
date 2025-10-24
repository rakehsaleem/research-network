"""
Knowledge Distillation Loss Functions for ViT Expert Distillation

This module implements the three core loss functions needed for Knowledge Distillation:
1. DistillationLoss: KL Divergence between teacher and student saliency predictions
2. AttentionAlignmentLoss: MSE between intermediate feature representations
3. HardLabelLoss: NSS or Cross-Entropy with ground truth saliency maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss using KL Divergence.
    
    Measures the difference between teacher and student saliency predictions
    using temperature-scaled softmax distributions.
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax scaling (higher = softer distributions)
            alpha: Weight for distillation loss vs hard labels
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self, 
        student_saliency: torch.Tensor, 
        teacher_saliency: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss between teacher and student predictions.
        
        Args:
            student_saliency: Student model saliency prediction [B, 1, H, W]
            teacher_saliency: Teacher model saliency prediction [B, 1, H, W]
            
        Returns:
            Distillation loss value
        """
        # Flatten spatial dimensions
        student_flat = student_saliency.view(student_saliency.size(0), -1)
        teacher_flat = teacher_saliency.view(teacher_saliency.size(0), -1)
        
        # Apply temperature scaling
        student_soft = F.log_softmax(student_flat / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = self.kl_div(student_soft, teacher_soft)
        
        # Scale by temperature squared
        distillation_loss = kl_loss * (self.temperature ** 2)
        
        return distillation_loss


class AttentionAlignmentLoss(nn.Module):
    """
    Attention Alignment Loss for intermediate feature matching.
    
    Aligns student model features with teacher model features using MSE loss.
    Helps student learn the teacher's attention patterns.
    """
    
    def __init__(self, feature_maps: Optional[Dict[str, float]] = None):
        """
        Initialize attention alignment loss.
        
        Args:
            feature_maps: Dictionary mapping feature names to weights
        """
        super().__init__()
        self.feature_maps = feature_maps or {
            'visual_features': 1.0,
            'fused_features': 1.0,
            'patch_saliency': 0.5
        }
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        student_features: Dict[str, torch.Tensor], 
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention alignment loss between student and teacher features.
        
        Args:
            student_features: Dictionary of student intermediate features
            teacher_features: Dictionary of teacher intermediate features
            
        Returns:
            Attention alignment loss value
        """
        total_loss = 0.0
        num_features = 0
        
        for feature_name, weight in self.feature_maps.items():
            if feature_name in student_features and feature_name in teacher_features:
                student_feat = student_features[feature_name]
                teacher_feat = teacher_features[feature_name]
                
                # Handle dimension mismatch by adaptive pooling
                if student_feat.shape != teacher_feat.shape:
                    # Adaptive pooling to match dimensions
                    if len(student_feat.shape) == 4:  # [B, C, H, W]
                        teacher_feat = F.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])
                    elif len(student_feat.shape) == 3:  # [B, seq_len, dim]
                        teacher_feat = F.adaptive_avg_pool1d(teacher_feat.transpose(1, 2), student_feat.shape[1]).transpose(1, 2)
                
                # Compute MSE loss
                feature_loss = self.mse_loss(student_feat, teacher_feat)
                total_loss += weight * feature_loss
                num_features += 1
        
        # Average over number of features
        if num_features > 0:
            return total_loss / num_features
        else:
            return torch.tensor(0.0, device=student_features[list(student_features.keys())[0]].device)


class HardLabelLoss(nn.Module):
    """
    Hard Label Loss using ground truth saliency maps.
    
    Supports multiple loss types: NSS (Normalized Scanpath Saliency),
    Cross-Entropy, and MSE for saliency prediction.
    """
    
    def __init__(self, loss_type: str = 'nss', reduction: str = 'mean'):
        """
        Initialize hard label loss.
        
        Args:
            loss_type: Type of loss ('nss', 'ce', 'mse', 'bce')
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        
        if self.loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == 'bce':
            self.loss_fn = nn.BCELoss(reduction=reduction)
        elif self.loss_type == 'nss':
            self.loss_fn = self._nss_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _nss_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Normalized Scanpath Saliency (NSS) loss.
        
        Args:
            prediction: Predicted saliency map [B, 1, H, W]
            target: Ground truth saliency map [B, 1, H, W]
            
        Returns:
            NSS loss value
        """
        # Flatten spatial dimensions
        pred_flat = prediction.view(prediction.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Normalize predictions
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        pred_std = pred_flat.std(dim=1, keepdim=True)
        pred_norm = (pred_flat - pred_mean) / (pred_std + 1e-8)
        
        # Compute NSS
        nss = (pred_norm * target_flat).sum(dim=1)
        
        if self.reduction == 'mean':
            return -nss.mean()
        elif self.reduction == 'sum':
            return -nss.sum()
        else:
            return -nss
    
    def forward(
        self, 
        student_saliency: torch.Tensor, 
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard label loss.
        
        Args:
            student_saliency: Student model saliency prediction [B, 1, H, W]
            ground_truth: Ground truth saliency map [B, 1, H, W]
            
        Returns:
            Hard label loss value
        """
        if self.loss_type == 'nss':
            return self.loss_fn(student_saliency, ground_truth)
        else:
            return self.loss_fn(student_saliency, ground_truth)


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss function for Knowledge Distillation training.
    
    Combines distillation loss, attention alignment loss, and hard label loss
    with configurable weights.
    """
    
    def __init__(
        self,
        distillation_weight: float = 1.0,
        attention_weight: float = 0.5,
        hard_label_weight: float = 0.3,
        temperature: float = 3.0,
        hard_loss_type: str = 'nss'
    ):
        """
        Initialize combined distillation loss.
        
        Args:
            distillation_weight: Weight for distillation loss
            attention_weight: Weight for attention alignment loss
            hard_label_weight: Weight for hard label loss
            temperature: Temperature for distillation loss
            hard_loss_type: Type of hard label loss
        """
        super().__init__()
        
        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.attention_loss = AttentionAlignmentLoss()
        self.hard_label_loss = HardLabelLoss(loss_type=hard_loss_type)
        
        self.distillation_weight = distillation_weight
        self.attention_weight = attention_weight
        self.hard_label_weight = hard_label_weight
        
    def forward(
        self,
        student_saliency: torch.Tensor,
        teacher_saliency: torch.Tensor,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_saliency: Student model saliency prediction
            teacher_saliency: Teacher model saliency prediction
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            ground_truth: Ground truth saliency map (optional)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Distillation loss
        dist_loss = self.distillation_loss(student_saliency, teacher_saliency)
        loss_dict['distillation'] = dist_loss
        
        # Attention alignment loss
        att_loss = self.attention_loss(student_features, teacher_features)
        loss_dict['attention'] = att_loss
        
        # Hard label loss (if ground truth provided)
        if ground_truth is not None:
            hard_loss = self.hard_label_loss(student_saliency, ground_truth)
            loss_dict['hard_label'] = hard_loss
        else:
            hard_loss = torch.tensor(0.0, device=student_saliency.device)
            loss_dict['hard_label'] = hard_loss
        
        # Combine losses
        total_loss = (
            self.distillation_weight * dist_loss +
            self.attention_weight * att_loss +
            self.hard_label_weight * hard_loss
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data
    batch_size = 2
    height, width = 224, 224
    
    student_saliency = torch.randn(batch_size, 1, height, width)
    teacher_saliency = torch.randn(batch_size, 1, height, width)
    ground_truth = torch.randn(batch_size, 1, height, width)
    
    # Dummy features
    student_features = {
        'visual_features': torch.randn(batch_size, 196, 768),
        'fused_features': torch.randn(batch_size, 196, 512)
    }
    teacher_features = {
        'visual_features': torch.randn(batch_size, 196, 768),
        'fused_features': torch.randn(batch_size, 196, 512)
    }
    
    # Test individual losses
    distillation_loss = DistillationLoss()
    attention_loss = AttentionAlignmentLoss()
    hard_label_loss = HardLabelLoss()
    
    dist_loss = distillation_loss(student_saliency, teacher_saliency)
    att_loss = attention_loss(student_features, teacher_features)
    hard_loss = hard_label_loss(student_saliency, ground_truth)
    
    print(f"Distillation Loss: {dist_loss.item():.4f}")
    print(f"Attention Loss: {att_loss.item():.4f}")
    print(f"Hard Label Loss: {hard_loss.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedDistillationLoss()
    total_loss, loss_dict = combined_loss(
        student_saliency, teacher_saliency, 
        student_features, teacher_features, 
        ground_truth
    )
    
    print(f"\nCombined Loss Components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
