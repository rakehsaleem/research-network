# ViT Expert Distillation for Real-time Drone Deployment

## Project Overview

This project implements an Expert Attention Prediction model using multimodal Knowledge Distillation for real-time drone deployment. The system uses a Teacher-Student architecture where a large multimodal Vision Transformer (ViT) teacher model distills knowledge to a lightweight CNN student model suitable for edge deployment on drones.

## Architecture

### Teacher Model (`MultimodalTeacherViT`)
- **Input**: Video features + Gaze history
- **Components**:
  - ViT Backbone for visual feature extraction
  - LSTM/GRU Layer for temporal gaze modeling
  - Multimodal Fusion Block for combining visual and gaze features
  - Saliency Prediction Head for attention map generation
- **Output**: Saliency map + Intermediate feature maps

### Student Model (`LightweightStudentCNN`)
- **Input**: Single video frame
- **Components**:
  - MobileNetV2 or similar lightweight backbone
  - Efficient saliency prediction head
- **Output**: Saliency map only

## Knowledge Distillation Strategy

The training process uses three key loss components:

1. **Distillation Loss**: KL Divergence between teacher and student saliency predictions
2. **Attention Alignment Loss**: MSE between intermediate feature representations
3. **Hard Label Loss**: NSS or Cross-Entropy with ground truth saliency maps

## Data Requirements

### Required Data Structure
```
01_data/
├── raw/                          # Raw video and gaze data
├── processed/
│   ├── video_features/          # Extracted video features
│   ├── gaze_features/           # Processed gaze data
│   └── gt_saliency_maps/        # Ground truth saliency maps
```

### Data Format
- **Video**: RGB frames at 30fps, resolution 224x224 or higher
- **Gaze**: Eye-tracking coordinates with timestamps
- **Saliency**: Ground truth attention maps (224x224)

## Project Structure

```
.
├── 01_data/                      # Data storage and processing
├── 02_models/                    # Model architectures
│   ├── teacher_model/           # Multimodal ViT teacher
│   └── student_model/           # Lightweight CNN student
├── 03_training/                 # Training scripts and losses
├── 04_deployment/               # Drone interface and hardware config
├── 05_analysis/                 # Evaluation and analysis tools
└── notebooks/                   # Jupyter notebooks for exploration
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place raw video and gaze data in `01_data/raw/`
   - Run preprocessing scripts to generate features

3. **Train Teacher Model**:
   ```bash
   python 03_training/train_teacher.py
   ```

4. **Distill to Student**:
   ```bash
   python 03_training/distillation_train.py
   ```

5. **Deploy on Drone**:
   ```bash
   python 04_deployment/drone_interface/deploy.py
   ```

## Key Features

- **Multimodal Fusion**: Combines visual and gaze information effectively
- **Real-time Inference**: Optimized student model for edge deployment
- **Knowledge Distillation**: Efficient transfer from teacher to student
- **Drone Integration**: Ready-to-deploy interface for UAV systems

## Performance Targets

- **Teacher Model**: High accuracy, multimodal understanding
- **Student Model**: <50ms inference time, <100MB model size
- **Deployment**: Real-time processing on drone hardware

## Contributing

This project is designed for research in attention prediction and knowledge distillation for autonomous systems. Contributions are welcome for improving model efficiency and deployment capabilities.