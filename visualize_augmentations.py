import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os
import random
from sklearn.cluster import KMeans
from config import TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, TRAIN_IMG_DIR, TRAIN_OUTPUT_DIR

def get_dominant_bg_color(image, k=3):
    """
    Detects the dominant background color using KMeans clustering.
    """
    pixels = image.reshape((-1, 3))  # Reshape to a list of pixels

    # Ensure K is not larger than the number of unique colors
    unique_colors = np.unique(pixels, axis=0)
    k = min(k, len(unique_colors))  # Reduce K if needed

    if k <= 1:
        return tuple(map(int, unique_colors[0]))  # If one unique color, return it

    # Apply KMeans
    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[kmeans.labels_[0]]  # Most common color
        return tuple(map(int, dominant_color))  # Convert to tuple of integers
    except Exception as e:
        print(f"Error in KMeans clustering: {e}")
        return (255, 255, 255)  # Default to white if clustering fails


def replace_black_with_bg(augmented_image, bg_color, tolerance=5):
    """
    Replace black pixels (or near black within tolerance) with the background color.
    """
    # Convert image to NumPy array if not already
    augmented_image = np.array(augmented_image, dtype=np.uint8)
    
    # Create a mask for nearly black pixels within the tolerance range
    mask = (augmented_image[..., 0] <= tolerance) & \
           (augmented_image[..., 1] <= tolerance) & \
           (augmented_image[..., 2] <= tolerance)

    # Replace black (or near black) pixels with the detected background color
    augmented_image[mask] = bg_color
    
    return augmented_image


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
        output = cv2.imread(output_path)
        if output is not None:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Original dimensions for reference
    original_height, original_width = image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")

    # Detect background color from the original image
    bg_color = get_dominant_bg_color(image)
    print(f"Detected background color: {bg_color}")

    # Ensure consistent augmentation by using the same transforms for inputs and outputs
    # And make sure they maintain the original dimensions
    augmentations = [
        ("vertical_flip", A.Compose([
            A.VerticalFlip(p=1.0)
        ], additional_targets={'output': 'image'})),
        
        ("HorizontalFlip", A.Compose([
            A.HorizontalFlip(p=1.0),
        ], additional_targets={'output': 'image'})),

        ("HorizontalFlip_and_vertical_flip", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ], additional_targets={'output': 'image'})),
    ]

    # Create grid of images
    num_augmentations = len(augmentations)
    rows = num_augmentations + 1  # Original + individual augmentations
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
    
    # Set a seed for consistent results
    random.seed(42)
    np.random.seed(42)
    
    # Create a directory for individual augmented images
    augmented_images_dir = os.path.join(output_dir, "individual_images")
    os.makedirs(augmented_images_dir, exist_ok=True)
    
    for i, (name, transform) in enumerate(augmentations):
        # Create a dictionary with inputs for augmentation
        aug_data = {"image": image}
        
        # Add output to augmentation if available
        if output is not None:
            aug_data["output"] = output
        
        # Apply the augmentations
        augmented = transform(**aug_data)
        
        # Get the augmented input and apply background replacement
        augmented_input = augmented["image"]
        augmented_input = replace_black_with_bg(augmented_input, bg_color)
        
        # Get the augmented output but DON'T replace background color (keep black)
        if output is not None:
            augmented_output = augmented["output"]
            # Do not apply color changes to output/mask - only geometric transformations
        else:
            augmented_output = None
        
        # Save individual augmented images with original dimensions
        input_filename = f"augmented_input_{name}.png"
        cv2.imwrite(
            os.path.join(augmented_images_dir, input_filename),
            cv2.cvtColor(augmented_input, cv2.COLOR_RGB2BGR)
        )
        
        if augmented_output is not None:
            output_filename = f"augmented_output_{name}.png"
            cv2.imwrite(
                os.path.join(augmented_images_dir, output_filename),
                cv2.cvtColor(augmented_output, cv2.COLOR_RGB2BGR)
            )
        
        # Display augmented input in the grid visualization
        plt.subplot(rows, cols, (i+2)*cols - 1)
        plt.imshow(augmented_input)
        plt.title(f"Input: {name}")
        plt.axis("off")
        
        # Display augmented output if available
        if augmented_output is not None:
            plt.subplot(rows, cols, (i+2)*cols)
            plt.imshow(augmented_output)
            plt.title(f"Output: {name}")
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentations_grid.png"), 
                facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Augmentation visualizations saved to {output_dir}")
    print(f"Individual augmented images saved to {augmented_images_dir} with original dimensions ({original_width}x{original_height})")

if __name__ == "__main__":
    visualize_augmentations()