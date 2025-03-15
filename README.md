# Garment Part Detection System - Setup & Usage Guide

This guide will help you set up and use the garment part detection system, specifically tailored to your directory structure with `/images`, `/masks`, and `/outputs` folders.

## Quick Start

```bash
# Clone or create project
# Install requirements
pip install -r requirements.txt

# Create directory structure and make run.sh executable 
chmod +x run.sh
./run.sh help

# Start training
./run.sh train

# Run inference on test images
./run.sh infer
```

## 1. Project Setup

### Directory Structure

The system expects data to be organized as follows:

```
data/
├── train/
│   ├── images/    # Input training images
│   ├── masks/     # Optional input masks (auxiliary information)
│   └── outputs/   # Ground truth segmentation masks (RGB format)
├── val/
│   ├── images/    # Input validation images
│   ├── masks/     # Optional input masks (auxiliary information)
│   └── outputs/   # Ground truth segmentation masks (RGB format)
└── test/          # Test images (no masks or outputs needed)
```

### Installation

1. **Clone the repository or create the structure yourself:**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make run.sh executable:**
   ```bash
   chmod +x run.sh
   ```

4. **Verify installation:**
   ```bash
   ./run.sh help
   ```

## 2. Data Preparation

### Expected Data Format

1. **Input Images:**
   - Standard RGB images (JPG, PNG)
   - Will be resized to dimensions defined in `config.py` (default: 768×1024)
   - Place in `data/train/images/` and `data/val/images/`

2. **Mask Images (Optional):**
   - RGB images used as auxiliary input if `useMaskForTraining = True` in `config.py`
   - Place in `data/train/masks/` and `data/val/masks/`

3. **Output Segmentation Masks (Ground Truth):**
   - RGB images with specific colors for each garment part:
     - Left Arm: RGB(22,22,22)
     - Right Arm: RGB(21,21,21)
     - Chest/Middle: RGB(5,5,5)
     - Collar (Front): RGB(24,24,24)
     - Body Back Parts: RGB(25,25,25)
     - Background: RGB(0,0,0)
   - Place in `data/train/outputs/` and `data/val/outputs/`

### Color Mapping

If your output segmentation masks use different RGB values, you should update the `CLASS_MAPPING` in `config.py` to match your specific color coding:

```python
CLASS_MAPPING = {
    # Label: RGB values - update these to match your actual colors
    0: (0, 0, 0),       # Background
    1: (22, 22, 22),    # Left Arm
    2: (21, 21, 21),    # Right Arm
    3: (5, 5, 5),       # Chest/Middle
    4: (24, 24, 24),    # Collar (Front)
    5: (25, 25, 25),    # Body Back Parts
}
```

## 3. Training the Model

### Basic Training

To start training with default settings:

```bash
./run.sh train
```

### Custom Training Options

```bash
./run.sh train --batch-size 8 --epochs 100 --lr 0.0001 --architecture deeplabv3plus --encoder resnet50
```

Available options:
- `--batch-size`: Number of images per batch (default from `config.py`)
- `--epochs`: Maximum number of training epochs
- `--lr`: Learning rate
- `--scratch`: Force starting training from scratch
- `--architecture`: Model architecture (options: unet, unet++, deeplabv3, deeplabv3plus, fpn, pspnet, hrnet)
- `--encoder`: Encoder backbone (options: resnet50, efficientnet-b0 through b7, resnext101_32x8d, etc.)

### Using Masks as Auxiliary Input

If you want to use the masks in `data/train/masks/` as auxiliary input during training:

1. Set `useMaskForTraining = True` in `config.py`
2. Ensure the system will still learn to predict the outputs in `data/train/outputs/`

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=logs
```

This provides visualizations of training/validation loss, IoU metrics, and sample predictions.

## 4. Running Inference

To predict segmentation for test images:

```bash
./run.sh infer --model models/best_model.pth --input data/test --output testResults
```

Options:
- `--model`: Path to model checkpoint (default: models/best_model.pth)
- `--input`: Directory containing test images (default: data/test)
- `--output`: Directory to save results (default: testResults)
- `--architecture`: Model architecture to use
- `--encoder`: Encoder backbone to use
- `--no-viz`: Disable visualization overlays

### Inference Output

For each test image, the system generates:
- `{filename}_mask.png`: Segmentation mask with class indices
- `{filename}_colored.png`: Colored visualization of segmentation
- `{filename}_overlay.png`: Original image with segmentation overlay

## 5. Evaluating Model Performance

To evaluate model performance on validation data:

```bash
./run.sh test --model models/best_model.pth
```

Options:
- `--model`: Path to model checkpoint
- `--output`: Directory to save evaluation results
- `--img-dir`: Directory with validation images (default: data/val/images)
- `--mask-dir`: Directory with ground truth masks (default: data/val/outputs)

### Evaluation Metrics

The system calculates:
- Mean IoU (Intersection over Union)
- Per-class IoU
- Precision, Recall, and F1 Score for each class
- Visual examples of predictions vs. ground truth

## 6. Configuration

The main configuration file `config.py` lets you customize:

- **Training parameters**: batch size, learning rate, epochs
- **Model architecture**: See `utils/model.py` for available options
- **Class mapping**: Map RGB values to class indices
- **Directory paths**: Update if your data is in different locations
- **Image resolution**: Default is 768×1024, change if needed
- **Optimization settings**: Early stopping, auto-recovery

## 7. Troubleshooting

### Common Issues

1. **"CUDA out of memory" error:**
   - Reduce batch size with `--batch-size 2` or even `--batch-size 1`
   - Try a smaller model with `--architecture unet --encoder efficientnet-b0`

2. **Poor segmentation quality:**
   - Check RGB color mapping in `config.py` matches your segmentation masks
   - Ensure proper data in `/outputs` directory
   - Try different model architectures and encoders

3. **Training not improving:**
   - Check learning rate (try `--lr 0.0001` or `--lr 0.00001`)
   - Increase training epochs with `--epochs 150`
   - Check that your ground truth segmentations are correct

4. **Masks not used in training:**
   - Verify `useMaskForTraining = True` in `config.py`
   - Check mask filenames match image filenames

### Debugging

- Check TensorBoard logs
- Look for printed error messages
- Verify dataset paths and masks are loaded correctly