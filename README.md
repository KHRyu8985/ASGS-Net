# Anatomy-Guided Network for Enhanced Boundary Refinement for Robust Segmentation

## Overview

This repository contains the implementation of ASGS-Net (Anatomy-guided Segmentation Network), a deep learning framework for automated segmentation of visceral adipose tissue (VAT) in PET/CT images. The proposed network leverages anatomical priors from TotalSegmentator to enhance boundary refinement and improve segmentation accuracy.

## Network Architecture

### ASGS-Net Overview
- **Backbone**: SegResNet-based architecture
- **Input**: 3D patches (128×128×128) from CT images
- **Key Components**:
  - Multi-level anatomical guidance module
  - Residual blocks for feature extraction
  - Sliding window merge strategy for full volumetric reconstruction

### Anatomical Guidance Module
1. **Contextual Anatomical Information Encoding**
   - Input: 8 anatomical segmentations (background, body trunk, torso fat, liver, heart, thoracic/lumbar vertebrae, ribs)
   - Processing: One-hot encoding → Gaussian smoothing → 3×3 convolutions with ReLU
   
2. **Feature Aggregation**
   - Spatially-Adaptive Normalization (SPADE)
   - Learned affine transformations (γ, β)
   - Multi-resolution feature modulation

## Installation

### 1. Environment Setup

```bash
# Install Pipenv
pip install pipenv

# Clone repository
git clone [repository_url]
cd viceral_fat_pl

# Install dependencies
pipenv install

# Activate environment
pipenv shell
```

### 2. Data Management

Currently, data is not shared due to privacy reason. But you can use public dataset with different segmentation task for the same purpose.

## Usage

### Training

The training can be done using either configuration file or command line arguments:

1. Using command line arguments:
```bash
# Train ASGS-Net (proposed model)
python scripts/proposed_train.py --arch_name SPADESegResNet --loss_fn DiceLoss --max_epochs 400 --check_val_every_n_epoch 10 --fold_number 1

# Train baseline model
python scripts/train.py --arch_name UNETR --loss_fn DiceLoss --max_epochs 300 --check_val_every_n_epoch 10 --fold_number 1
```

### Inference

```bash
# Inference with ASGS-Net
python scripts/proposed_train.py --config config/config.yaml --checkpoint_path path/to/checkpoint.ckpt

# Inference with baseline model
python scripts/train.py --config config/baseline_config.yaml --checkpoint_path path/to/checkpoint.ckpt
```

## Project Structure

```
viceral_fat_pl/\
├── data/          # Dataset storage
│   └── data_splits.yaml # Data split configuration
├── nbs/           # Jupyter notebooks
├── results/       # Experimental results
├── scripts/       # Training and evaluation scripts
│   ├── proposed_train.py # ASGS-Net training script
│   └── train.py   # Baseline model training script
├── src/           # Source code
│   ├── data/      # Data processing modules
│   ├── losses/    # Loss functions
│   ├── metrics/   # Evaluation metrics
│   ├── models/    # Model implementations
│   └── utils/     # Utility functions
└── weights/       # Model weights
```

## Contact

For questions and issues, please contact Kanghyun Ryu (khryu@kist.re.kr)


