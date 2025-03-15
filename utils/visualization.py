import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from config import CLASS_MAPPING

def visualize_prediction(image, true_mask=None, pred_mask=None, alpha=0.5):
    """
    Visualize the prediction result.
    
    Args:
        image: Original RGB image
        true_mask: Ground truth mask (optional)
        pred_mask: Predicted mask (optional)
        alpha: Transparency factor for the overlay
        
    Returns:
        result: Visualization image with original and masks
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Scale to 0-255 if normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
        
    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.dim() > 2:  # If pred_mask is one-hot or has channel dimension
            pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
        else:
            pred_mask = pred_mask.cpu().numpy()
    
    # Create color mapping for visualization
    color_map = {}
    for class_idx, rgb in CLASS_MAPPING.items():
        color_map[class_idx] = rgb
    
    # Create the result visualization
    if true_mask is not None and pred_mask is not None:
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Plot ground truth mask
        true_mask_rgb = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            true_mask_rgb[true_mask == class_idx] = color
        axes[1].imshow(true_mask_rgb)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        # Plot predicted mask
        pred_mask_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            pred_mask_rgb[pred_mask == class_idx] = color
        axes[2].imshow(pred_mask_rgb)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
        
        # Tight layout
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        result = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
    elif pred_mask is not None:
        # Create overlay of prediction on original image
        pred_mask_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            pred_mask_rgb[pred_mask == class_idx] = color
            
        # Create blended overlay
        result = cv2.addWeighted(image, 1 - alpha, pred_mask_rgb, alpha, 0)
        
    else:
        # Just return the original image
        result = image.copy()
    
    return result

def save_colored_mask(mask, output_path, class_mapping=CLASS_MAPPING):
    """
    Save a colored mask based on class indices
    
    Args:
        mask: Single-channel class index mask
        output_path: Path to save the colored mask
        class_mapping: Dictionary mapping class indices to RGB values
    """
    # Create colored mask
    height, width = mask.shape[:2]
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign colors based on class indices
    for class_idx, rgb in class_mapping.items():
        colored_mask[mask == class_idx] = rgb
    
    # Save the colored mask
    cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
def create_confusion_visualization(image, mask, prediction):
    """
    Create a visualization showing correct/incorrect predictions
    
    Args:
        image: Original RGB image
        mask: Ground truth mask
        prediction: Predicted mask
        
    Returns:
        result: Visualization of confusion
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Scale to 0-255 if normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    if isinstance(prediction, torch.Tensor):
        if prediction.dim() > 2:
            prediction = torch.argmax(prediction, dim=0).cpu().numpy()
        else:
            prediction = prediction.cpu().numpy()
    
    # Create confusion visualization
    confusion = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Correct predictions (green)
    correct = (mask == prediction)
    confusion[correct] = [0, 255, 0]
    
    # Incorrect predictions (red)
    incorrect = ~correct
    confusion[incorrect] = [255, 0, 0]
    
    # Blend with original image
    result = cv2.addWeighted(image, 0.7, confusion, 0.3, 0)
    
    return result