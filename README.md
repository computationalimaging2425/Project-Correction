


# Project-Correction: Performance Analysis for Projection-Correction Methods in Motion Deblurring Problems

This repository contains the implementation and analysis of projection-correction methods for motion deblurring problems using a diffusion model.

## Overview

This project implements and compares two state-of-the-art projection-correction methods for solving inverse problems in image deblurring:

1. **DPS (Diffusion Posterior Sampling)** - A method that combines data fidelity with diffusion model priors during reverse sampling

2. **RED-Diff (Regularization by Denoising Diffusion)** - An optimization-based method that uses gradient-based optimization to reconstruct images by minimizing a combined objective

## Dataset

The project uses the **Mayo Clinic CT Dataset** of low-dose CT scans.

## Data Augmentation

The project implements comprehensive data augmentation techniques to improve model robustness:

- **Fixed rotations**
- **Horizontal flip**
- **Gaussian noise**
- **Salt-and-pepper noise**
- **Brightness adjustment**
- **Contrast adjustment**

## Architecture

The project uses a UNet2D model from the Diffusers library with the following configuration:

- Sample size: 128×128 pixels
- Input/Output channels: 1 (grayscale)
- Block output channels: (64, 128, 256)
- Includes attention mechanisms in deeper layers
- Dropout: 0.1 for regularization

## Installation

### Prerequisites

Choose the appropriate requirements file based on your system:

- **Ubuntu CPU**: `requirements_ubuntu_cpu.txt`
- **Ubuntu GPU (CUDA 12.8)**: `requirements_ubuntu_gpu_cuda128.txt`
- **Windows GPU (CUDA 12.8)**: `requirements_windows_gpu_cuda128.txt`
- **macOS**: `requirements_mac.txt`

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/computationalimaging2425/Project-Correction.git
cd Project-Correction

# Install dependencies (example for Ubuntu CPU)
pip install -r requirements_ubuntu_cpu.txt
```

## Key Features

### Environment Setup

The project supports both local and Google Colab environments with automatic path configuration

### Model Training and Checkpointing

- **Checkpoint saving**: Saves model and optimizer states with epoch information
- **Checkpoint loading**: Supports loading pre-trained models with state dict key handling

### Sampling and Validation

- **Pure noise sampling**: Generates images from random noise using DDIM scheduler
- **Validation reconstructions**: Evaluates model performance on test data with one-step reconstruction

### Projection-Correction Methods

#### DPS Implementation
The DPS method modifies the standard DDIM reverse process to incorporate measurement consistency through posterior correction

#### RED-Diff Implementation  
The RED-Diff method uses optimization-based reconstruction with configurable weighting strategies (linear, sqrt, square, log, clip, const)

### Results
We achieved excellent result, obtainining 40db PSNR and 0.984 SSIM values for the reconstructed images.
Further details are in the report directory.

### Notes
- The project includes comprehensive data augmentation to improve model robustness with 8 different augmentation types applied to each base image
- Both DPS and RED-Diff methods include visualization functions that automatically compute and display PSNR and SSIM metrics for performance evaluation
- The implementation supports both CPU and GPU execution with automatic device detection and model placement
- Environment variables can be used to customize data paths and model directories for different deployment scenarios
