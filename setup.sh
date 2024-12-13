#!/bin/bash

# Create and activate conda environment
conda create -n style_classifier_2 python=3.9 -y
conda activate style_classifier_2

# Install PyTorch and related packages
pip install torch torchvision torchaudio

# Install additional dependencies
pip install numpy pandas matplotlib pillow tqdm

# Verify setup
python src/setup.py 