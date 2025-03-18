import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os
import random
from config import TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, TRAIN_IMG_DIR, TRAIN_OUTPUT_DIR

def visualize_augmentations(output_dir="augmentation_results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a random image from training folder
    image_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {TRAIN_IMG_DIR}")
        return
        
    random_image = random.choice(image_files)
    image_path = os.path.join(TRAIN_IMG_DIR, random_image)
    print(f"Selected random image: {image_path}")

    # Try to find corresponding output/mask file
    output_path = None
    if os.path.exists(TRAIN_OUTPUT_DIR):
        output_file = random_image  # Assuming same filename in output directory
        potential_output_path = os.path.join(TRAIN_OUTPUT_DIR, output_file)
        if os.path.exists(potential_output_path):
            output_path = potential_output_path
            print(f"Found corresponding output: {output_path}")

    # Read the input image (cloth)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read the output/mask
    output = None
    if output_path:
        output = cv2.imread(output_path)  # Keep the output mask as-is
    
    # Original dimensions for reference
    original_height, original_width = image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")

    # Augmentations including fixed MotionBlur (blur_limit=(3, 9))
    augmentations = [

        ("Rotate & Shear & Zoom & Blur & Noise & Brightness & Contrast", A.Compose([
            A.Affine(rotate=(-40, 40), shear=(-25, 25), scale=(0.7, 1.3), cval=246, p=1.0),
            A.GaussianBlur(blur_limit=(5, 9), p=0.8),
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.2, 0.5), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.7),
            A.MotionBlur(blur_limit=10, p=0.8, centered=False),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.CoarseDropout(max_holes=10, max_height=20, max_width=20, p=0.8)
        ])),

        ("Elastic & Grid Distortion & Brightness & Sharpen & Flip & Solarize & Invert", A.Compose([
            A.ElasticTransform(alpha=3, sigma=60, alpha_affine=60, p=1.0),
            A.GridDistortion(num_steps=6, distort_limit=0.4, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.7),
            A.Sharpen(alpha=(0.3, 0.7), lightness=(0.6, 1.2), p=0.7),
            A.VerticalFlip(p=0.7),
            A.Solarize(threshold=100, p=0.8),
            A.InvertImg(p=0.7)
        ])),

        ("Flip & Solarize & Noise & Scale & Rotate & Dropout & Equalize", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Solarize(threshold=110, p=0.9),
            A.ISONoise(color_shift=(0.03, 0.07), intensity=(0.3, 0.6), p=0.7),
            A.RandomScale(scale_limit=0.3, p=0.7),
            A.Rotate(limit=35, p=0.8),
            A.PixelDropout(dropout_prob=0.15, per_channel=True, p=0.8),
            A.Equalize(mode='cv', by_channels=True, p=0.8)
        ])),

        ("Affine & Defocus & Sharpen & Contrast & Pixel Dropout & Translate & Color Jitter", A.Compose([
            A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), shear=(-10, 10), cval=246, p=1.0),
            A.Defocus(radius=(5, 9), p=0.8),
            A.Sharpen(alpha=(0.3, 0.7), lightness=(0.6, 1.2), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.PixelDropout(dropout_prob=0.15, per_channel=True, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7)
        ])),

        ("Rotate & Flip & Grid Distortion & Noise & Sharpen & Emboss & Perspective", A.Compose([
            A.Rotate(limit=50, p=0.8),
            A.HorizontalFlip(p=0.7),
            A.GridDistortion(num_steps=6, distort_limit=0.4, p=0.8),
            A.ISONoise(color_shift=(0.03, 0.09), intensity=(0.3, 0.6), p=0.7),
            A.Sharpen(alpha=(0.4, 0.8), lightness=(0.7, 1.3), p=0.7),
            A.Emboss(alpha=(0.2, 0.6), strength=(0.3, 1.0), p=0.8),
            A.Perspective(scale=(0.05, 0.15), p=0.7)
        ])),

        ("Elastic & Scale & Motion Blur & Contrast & Solarize & Cutout & Jitter", A.Compose([
            A.ElasticTransform(alpha=4, sigma=70, alpha_affine=70, p=1.0),
            A.RandomScale(scale_limit=0.35, p=0.7),
            A.MotionBlur(blur_limit=10, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.7),
            A.Solarize(threshold=120, p=0.8),
            A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7)
        ])),

        ("Perspective & Cutout & Noise & Blur & Sharpen & Flip & Pixel Dropout", A.Compose([
            A.Perspective(scale=(0.05, 0.15), p=0.9),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_height=10, min_width=10, p=0.8),
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.3, 0.6), p=0.7),
            A.GaussianBlur(blur_limit=7, p=0.7),
            A.Sharpen(alpha=(0.3, 0.6), lightness=(0.7, 1.2), p=0.8),
            A.HorizontalFlip(p=0.8),
            A.PixelDropout(dropout_prob=0.1, per_channel=True, p=0.8)
        ])),

        ("Invert & Flip & Scale & Defocus & Contrast & Shear & Brightness", A.Compose([
            A.InvertImg(p=1.0),
            A.HorizontalFlip(p=0.8),
            A.RandomScale(scale_limit=0.4, p=0.7),
            A.Defocus(radius=(3, 7), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.7),
            A.Affine(shear=(-15, 15), cval=255, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.8)
        ])),

        ("Jitter & Translate & Rotate & Dropout & Blur & Elastic & Noise", A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=30, p=0.9),
            A.PixelDropout(dropout_prob=0.1, per_channel=True, p=0.7),
            A.MotionBlur(blur_limit=8, p=0.7),
            A.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=0.8),
            A.ISONoise(color_shift=(0.02, 0.07), intensity=(0.3, 0.6), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7)
        ])),

        ("Equalize & Flip & Zoom & Shear & Noise & Solarize & Perspective", A.Compose([
            A.Equalize(mode='cv', by_channels=True, p=0.8),
            A.VerticalFlip(p=0.7),
            A.Affine(scale=(0.75, 1.25), shear=(-15, 15), cval=255, p=1.0),
            A.ISONoise(color_shift=(0.03, 0.08), intensity=(0.4, 0.6), p=0.7),
            A.Solarize(threshold=130, p=0.8),
            A.Perspective(scale=(0.03, 0.12), p=0.7),
            A.MotionBlur(blur_limit=10, p=0.8, centered=False)
        ]))
    ]

    # Create grid of images
    num_augmentations = len(augmentations)
    rows = num_augmentations + 2  # Original + individual + mixed
    cols = 2 if output is not None else 1  # Input and output if available
    
    plt.figure(figsize=(cols*6, rows*6))
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Display original image and mask
    plt.subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title(f"Original Input ({original_width}x{original_height})")
    plt.axis("off")
    
    if output is not None:
        plt.subplot(rows, cols, 2)
        plt.imshow(output)
        plt.title("Original Output")
        plt.axis("off")
    
    for i, (name, transform) in enumerate(augmentations):
        augmented = transform(image=image, mask=output if output is not None else image)
        augmented_input = augmented["image"]
        augmented_output = augmented["mask"] if output is not None else None

        plt.subplot(rows, cols, (i+1)*cols + 1)
        plt.imshow(augmented_input)
        plt.title(f"Input: {name}")
        plt.axis("off")
        
        if output is not None:
            plt.subplot(rows, cols, (i+1)*cols + 2)
            plt.imshow(augmented_output)
            plt.title(f"Output: {name}")
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentations_grid.png"), 
                facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Augmentation visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_augmentations()
