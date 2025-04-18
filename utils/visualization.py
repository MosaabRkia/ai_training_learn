# visualization.py
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

def visualize_binary_mask(image, mask, title="Binary Mask", alpha=0.5):
    """
    Specialized visualization for binary hand segmentation.
    
    Args:
        image: Original RGB image
        mask: Binary mask (0=background, 1=hand)
        title: Title for the visualization
        alpha: Transparency factor for the overlay
        
    Returns:
        result: Visualization image with hand segmentation highlighted
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Scale to 0-255 if normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Create colored mask for the hands
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # Use a distinctive color for hands (bright green)
    mask_rgb[mask == 1] = [0, 255, 0]
    
    # Create the blended result
    result = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    
    # Add a border around the hand segments for better visibility
    if np.any(mask == 1):
        # Create a kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        # Dilate the mask to get the boundary
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated - mask.astype(np.uint8)
        
        # Add boundary to the result in a contrasting color (e.g., red)
        result[boundary == 1] = [255, 0, 0]
    
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

def visualize_hand_segmentation_metrics(image, true_mask, pred_mask):
    """
    Create a comprehensive visualization specifically for hand segmentation evaluation.
    
    Args:
        image: Original RGB image
        true_mask: Ground truth mask (0=background, 1=hand)
        pred_mask: Predicted mask (0=background, 1=hand)
        
    Returns:
        result: Visualization showing original, prediction, ground truth, and confusion
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
        
    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.dim() > 2:
            pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
        else:
            pred_mask = pred_mask.cpu().numpy()
    
    # Calculate metrics
    true_positive = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    false_positive = np.logical_and(pred_mask == 1, true_mask == 0).sum()
    false_negative = np.logical_and(pred_mask == 0, true_mask == 1).sum()
    true_negative = np.logical_and(pred_mask == 0, true_mask == 0).sum()
    
    # Calculate IoU for hand class
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)
    
    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    
    # Create visualization with metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Ground truth mask
    mask_viz = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    mask_viz[true_mask == 1] = [0, 255, 0]  # Green for hands
    axes[0, 1].imshow(mask_viz)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis('off')
    
    # Prediction mask
    pred_viz = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    pred_viz[pred_mask == 1] = [0, 255, 0]  # Green for hands
    axes[1, 0].imshow(pred_viz)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis('off')
    
    # Confusion matrix visualization
    confusion = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    # True positives (green)
    confusion[np.logical_and(pred_mask == 1, true_mask == 1)] = [0, 255, 0]
    # False positives (red)
    confusion[np.logical_and(pred_mask == 1, true_mask == 0)] = [255, 0, 0]
    # False negatives (blue)
    confusion[np.logical_and(pred_mask == 0, true_mask == 1)] = [0, 0, 255]
    
    axes[1, 1].imshow(confusion)
    axes[1, 1].set_title(f"Error Analysis: IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    axes[1, 1].axis('off')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=(0/255, 255/255, 0/255), label='True Positive'),
        plt.Rectangle((0, 0), 1, 1, color=(255/255, 0/255, 0/255), label='False Positive'),
        plt.Rectangle((0, 0), 1, 1, color=(0/255, 0/255, 255/255), label='False Negative')
    ]
    axes[1, 1].legend(handles=legend_elements, loc='lower right')
    
    # Tight layout
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    result = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    
    return result