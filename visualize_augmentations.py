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
    
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read the output/mask if available
    output = None
    if output_path:
        output = cv2.imread(output_path)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    else:
        print("No matching output file found!")
        return
    
    # Original dimensions
    original_height, original_width = image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")
    
    # Add white padding first to avoid black borders after resize
    def add_white_padding(img, pad=50):
        h, w = img.shape[:2]
        white_bg = np.ones((h + 2*pad, w + 2*pad, 3), dtype=np.uint8) * 255
        white_bg[pad:pad+h, pad:pad+w] = img
        return white_bg
    
    # Add padding to both images
    image_padded = add_white_padding(image)
    output_padded = add_white_padding(output)
    
    # Define transformations that properly handle both image and mask
    transforms = [
        ("30% Size", A.Compose([
            A.Resize(height=int(original_height*0.3), width=int(original_width*0.3))
        ], additional_targets={'mask': 'image'})),
        
        ("200% Size", A.Compose([
            A.Resize(height=int(original_height*2.0), width=int(original_width*2.0))
        ], additional_targets={'mask': 'image'})),
        
        ("HorizontalFlip", A.Compose([
            A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
            A.HorizontalFlip(p=1.0)
        ], additional_targets={'mask': 'image'})),
        
        ("Rotation 30°", A.Compose([
            # Use PadIfNeeded to ensure white padding
            A.PadIfNeeded(min_height=int(TRAIN_IMAGE_HEIGHT*1.5), 
                          min_width=int(TRAIN_IMAGE_WIDTH*1.5),
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=[255, 255, 255]),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=1.0),
            A.CenterCrop(height=TRAIN_IMAGE_HEIGHT, width=TRAIN_IMAGE_WIDTH)
        ], additional_targets={'mask': 'image'})),
        
        ("Rotation 45°", A.Compose([
            # Use PadIfNeeded to ensure white padding
            A.PadIfNeeded(min_height=int(TRAIN_IMAGE_HEIGHT*1.5), 
                          min_width=int(TRAIN_IMAGE_WIDTH*1.5),
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=[255, 255, 255]),
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=1.0),
            A.CenterCrop(height=TRAIN_IMAGE_HEIGHT, width=TRAIN_IMAGE_WIDTH)
        ], additional_targets={'mask': 'image'})),
        
        ("Color Enhanced", A.Compose([
            A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.2, p=1.0)
        ], additional_targets={'mask': 'image'})),
    ]
    
    # Combined transformation
    combined_transform = A.Compose([
        A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        # Use PadIfNeeded to ensure white padding
        A.PadIfNeeded(min_height=int(TRAIN_IMAGE_HEIGHT*1.5), 
                     min_width=int(TRAIN_IMAGE_WIDTH*1.5),
                     border_mode=cv2.BORDER_CONSTANT, 
                     value=[255, 255, 255]),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=1.0),
        A.CenterCrop(height=TRAIN_IMAGE_HEIGHT, width=TRAIN_IMAGE_WIDTH),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7)
    ], additional_targets={'mask': 'image'})
    
    # Create grid
    rows = len(transforms) + 2  # Original + transforms + combined
    cols = 2  # Input and output
    
    # Create figure with white background
    plt.figure(figsize=(cols*6, rows*6))
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Display original images
    plt.subplot(rows, cols, 1)
    plt.gca().set_facecolor('white')
    plt.imshow(image)
    plt.title(f"Original Input ({original_width}x{original_height})")
    plt.axis("off")
    
    plt.subplot(rows, cols, 2)
    plt.gca().set_facecolor('white')
    plt.imshow(output)
    plt.title("Original Output")
    plt.axis("off")
    
    # Apply and display each transformation
    for i, (name, transform) in enumerate(transforms):
        # Apply same transformation to both input and mask
        augmented = transform(image=image_padded.copy(), mask=output_padded.copy())
        augmented_input = augmented['image']
        augmented_output = augmented['mask']
        
        h, w = augmented_input.shape[:2]
        
        # Display input
        plt.subplot(rows, cols, (i+1)*cols + 1)
        plt.gca().set_facecolor('white')
        plt.imshow(augmented_input)
        plt.title(f"Input: {name} ({w}x{h})")
        plt.axis("off")
        
        # Display output
        plt.subplot(rows, cols, (i+1)*cols + 2)
        plt.gca().set_facecolor('white')
        plt.imshow(augmented_output)
        plt.title(f"Output: {name}")
        plt.axis("off")
    
    # Apply and display combined transformation
    augmented = combined_transform(image=image_padded.copy(), mask=output_padded.copy())
    augmented_input = augmented['image']
    augmented_output = augmented['mask']
    
    h, w = augmented_input.shape[:2]
    
    plt.subplot(rows, cols, (rows-1)*cols + 1)
    plt.gca().set_facecolor('white')
    plt.imshow(augmented_input)
    plt.title(f"Input: Combined ({w}x{h})")
    plt.axis("off")
    
    plt.subplot(rows, cols, (rows-1)*cols + 2)
    plt.gca().set_facecolor('white')
    plt.imshow(augmented_output)
    plt.title("Output: Combined")
    plt.axis("off")
    
    # Save the grid with white background
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentations_grid.png"), 
                facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Augmentation visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_augmentations()