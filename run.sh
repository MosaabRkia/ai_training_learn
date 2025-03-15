#!/bin/bash

# Create required directories
mkdir -p data/train/images data/train/masks data/train/outputs
mkdir -p data/val/images data/val/masks data/val/outputs
mkdir -p data/test
mkdir -p models models_checkpoints logs testResults

# Function to display help
show_help() {
    echo "Garment Segmentation Training & Inference Script"
    echo "Usage:"
    echo "  ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train