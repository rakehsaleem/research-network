"""
Knowledge Distillation Training Script for ViT Expert Distillation

This script implements the main training loop for distilling knowledge from
a multimodal teacher ViT model to a lightweight student CNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.teacher_model.architecture import MultimodalTeacherViT
from models.student_model.architecture import LightweightStudentCNN
from training.losses import CombinedDistillationLoss


class DistillationTrainer:
    """
    Trainer class for Knowledge Distillation from Teacher to Student model.
    
    Handles the complete training pipeline including:
    - Model initialization and loading
    - Training loop with distillation losses
    - Validation and checkpointing
    - Logging and monitoring
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the distillation trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.teacher_model = None
        self.student_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        if config.get('use_wandb', False):
            wandb.init(project="vit-expert-distillation", config=config)
    
    def setup_models(self):
        """
        Initialize teacher and student models.
        
        Teacher model is loaded from checkpoint and frozen.
        Student model is initialized for training.
        """
        print("Setting up models...")
        
        # Initialize teacher model
        self.teacher_model = MultimodalTeacherViT(
            vit_model=self.config.get('teacher_vit_model', 'vit_base_patch16_224'),
            visual_dim=self.config.get('teacher_visual_dim', 768),
            gaze_dim=self.config.get('teacher_gaze_dim', 512),
            fusion_dim=self.config.get('teacher_fusion_dim', 512)
        )
        
        # Load teacher checkpoint if provided
        teacher_checkpoint = self.config.get('teacher_checkpoint')
        if teacher_checkpoint and os.path.exists(teacher_checkpoint):
            print(f"Loading teacher model from {teacher_checkpoint}")
            checkpoint = torch.load(teacher_checkpoint, map_location=self.device)
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set teacher to eval mode and freeze weights
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Initialize student model
        self.student_model = LightweightStudentCNN(
            backbone=self.config.get('student_backbone', 'mobilenet_v2'),
            width_mult=self.config.get('student_width_mult', 1.0),
            img_size=self.config.get('img_size', 224),
            pretrained=self.config.get('student_pretrained', True)
        )
        
        # Move models to device
        self.teacher_model = self.teacher_model.to(self.device)
        self.student_model = self.student_model.to(self.device)
        
        print(f"Teacher model parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"Student model parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print(f"Student model size: {self.student_model.get_model_size_mb():.2f} MB")
    
    def setup_optimizer_and_scheduler(self):
        """
        Initialize optimizer and learning rate scheduler for student model.
        """
        # Optimizer for student model only
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
    
    def setup_criterion(self):
        """
        Initialize the combined distillation loss function.
        """
        self.criterion = CombinedDistillationLoss(
            distillation_weight=self.config.get('distillation_weight', 1.0),
            attention_weight=self.config.get('attention_weight', 0.5),
            hard_label_weight=self.config.get('hard_label_weight', 0.3),
            temperature=self.config.get('temperature', 3.0),
            hard_loss_type=self.config.get('hard_loss_type', 'nss')
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        loss_components = {'distillation': 0.0, 'attention': 0.0, 'hard_label': 0.0}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            video_frames = batch['video_frame'].to(self.device)
            gaze_history = batch['gaze_history'].to(self.device)
            ground_truth = batch.get('ground_truth', None)
            if ground_truth is not None:
                ground_truth = ground_truth.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_saliency, teacher_features = self.teacher_model(video_frames, gaze_history)
            
            # Forward pass through student model
            student_saliency = self.student_model(video_frames)
            
            # Extract student features for attention alignment
            # Note: This is a simplified version - in practice, you'd modify the student model
            # to return intermediate features similar to the teacher model
            student_features = {
                'visual_features': torch.randn_like(teacher_features['visual_features']),  # Placeholder
                'fused_features': torch.randn_like(teacher_features['fused_features'])    # Placeholder
            }
            
            # Compute combined loss
            loss, loss_dict = self.criterion(
                student_saliency=student_saliency,
                teacher_saliency=teacher_saliency,
                student_features=student_features,
                teacher_features=teacher_features,
                ground_truth=ground_truth
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), 
                    self.config['grad_clip']
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/distillation_loss': loss_dict['distillation'].item(),
                    'train/attention_loss': loss_dict['attention'].item(),
                    'train/hard_label_loss': loss_dict['hard_label'].item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the student model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.student_model.eval()
        self.teacher_model.eval()
        
        total_loss = 0.0
        loss_components = {'distillation': 0.0, 'attention': 0.0, 'hard_label': 0.0}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Extract batch data
                video_frames = batch['video_frame'].to(self.device)
                gaze_history = batch['gaze_history'].to(self.device)
                ground_truth = batch.get('ground_truth', None)
                if ground_truth is not None:
                    ground_truth = ground_truth.to(self.device)
                
                # Forward pass through teacher model
                teacher_saliency, teacher_features = self.teacher_model(video_frames, gaze_history)
                
                # Forward pass through student model
                student_saliency = self.student_model(video_frames)
                
                # Extract student features (placeholder)
                student_features = {
                    'visual_features': torch.randn_like(teacher_features['visual_features']),
                    'fused_features': torch.randn_like(teacher_features['fused_features'])
                }
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    student_saliency=student_saliency,
                    teacher_saliency=teacher_saliency,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    ground_truth=ground_truth
                )
                
                # Accumulate losses
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key in loss_components:
                        loss_components[key] += value.item()
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.get('checkpoint_dir', '02_models/student_model/checkpoints'),
            f'checkpoint_epoch_{epoch}.pth'
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.get('checkpoint_dir', '02_models/student_model/checkpoints'),
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("Starting Knowledge Distillation Training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {self.config.get('num_epochs', 100)}")
        
        # Setup components
        self.setup_models()
        self.setup_optimizer_and_scheduler()
        self.setup_criterion()
        
        # Training loop
        for epoch in range(self.config.get('num_epochs', 100)):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            
            # Log metrics
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_metrics['total_loss'],
                    'train/distillation_loss': train_metrics['distillation'],
                    'train/attention_loss': train_metrics['attention'],
                    'train/hard_label_loss': train_metrics['hard_label'],
                    'val/total_loss': val_metrics['total_loss'],
                    'val/distillation_loss': val_metrics['distillation'],
                    'val/attention_loss': val_metrics['attention'],
                    'val/hard_label_loss': val_metrics['hard_label'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            if epoch % self.config.get('save_freq', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """
    Main function for running distillation training.
    """
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    
    # Model configuration
    parser.add_argument('--teacher_checkpoint', type=str, help='Path to teacher model checkpoint')
    parser.add_argument('--student_backbone', type=str, default='mobilenet_v2', help='Student model backbone')
    parser.add_argument('--student_width_mult', type=float, default=1.0, help='Student model width multiplier')
    
    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Loss configuration
    parser.add_argument('--distillation_weight', type=float, default=1.0, help='Distillation loss weight')
    parser.add_argument('--attention_weight', type=float, default=0.5, help='Attention loss weight')
    parser.add_argument('--hard_label_weight', type=float, default=0.3, help='Hard label loss weight')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='01_data', help='Data directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    # Logging and checkpointing
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--checkpoint_dir', type=str, default='02_models/student_model/checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = vars(args)
    
    # Initialize trainer
    trainer = DistillationTrainer(config)
    
    # Note: In a real implementation, you would load your actual data loaders here
    # For now, this is a placeholder structure
    print("Note: This is a template implementation.")
    print("You need to implement your data loading pipeline.")
    print("The training structure is ready for your dataset.")
    
    # Example of how to use the trainer:
    # train_loader = create_train_loader(config)
    # val_loader = create_val_loader(config)
    # trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
