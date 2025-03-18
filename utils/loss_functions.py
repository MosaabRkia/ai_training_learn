import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES
from tqdm import tqdm
import os
class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    def __init__(self, weight=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight  # Class weights
        
    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for class_idx in range(num_classes):
            class_probs = probs[:, class_idx, :, :]
            class_targets = targets_one_hot[:, class_idx, :, :]
            
            intersection = (class_probs * class_targets).sum(dim=(1, 2))
            union = class_probs.sum(dim=(1, 2)) + class_targets.sum(dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())
        
        # Apply class weights if provided
        if self.weight is not None:
            dice_scores = [w * d for w, d in zip(self.weight, dice_scores)]
        
        # Return 1 - mean dice score (loss)
        return 1 - torch.mean(torch.stack(dice_scores))


class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, logits, targets):
        # Get number of classes
        num_classes = logits.shape[1]
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=1)
        
        # Gather the target class probabilities
        targets_flat = targets.view(-1)
        log_probs_flat = log_probs.view(-1, num_classes)
        target_probs = log_probs_flat.gather(1, targets_flat.unsqueeze(1))
        
        # Calculate focal loss
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).view(-1)
        loss = -((1 - pt) ** self.gamma) * target_probs.view(-1)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets_flat)
            loss = alpha_weight * loss
            
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Dice Loss + Focal Loss
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, 
                 dice_smooth=1e-5, focal_gamma=2.0,
                 class_weights=None):
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize class weights if provided
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Initialize individual losses
        self.dice_loss = DiceLoss(weight=class_weights, smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        
    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


# def calculate_class_weights(dataset):
#     """
#     Calculate class weights based on class frequencies with safeguards
#     """
#     class_counts = torch.zeros(NUM_CLASSES)
    
#     # Count occurrences of each class
#     for i, (_, mask) in enumerate(dataset):
#         if i % 100 == 0:
#             print(f"Processing image {i}/{len(dataset)} for class weights...")
#         for class_idx in range(NUM_CLASSES):
#             class_counts[class_idx] += (mask == class_idx).sum().item()
    
#     # Add a small epsilon to prevent division by zero for classes that may not appear
#     epsilon = 1e-5
#     class_counts = class_counts + epsilon
    
#     # Calculate weights (inverse of frequency)
#     total_pixels = class_counts.sum()
#     class_weights = total_pixels / (class_counts * NUM_CLASSES)
    
#     # Normalize weights
#     class_weights = class_weights / class_weights.sum()
    
#     # Optional: Cap weights to avoid extremely high values
#     max_weight = 10.0
#     class_weights = torch.clamp(class_weights, max=max_weight)
    
#     # Move to the correct device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     class_weights = class_weights.to(device)
    
#     return class_weights

# def calculate_class_weights(dataset):
#     """
#     Calculate class weights based on class frequencies with GPU acceleration
#     """
#     # Initialize counts on GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     class_counts = torch.zeros(NUM_CLASSES, device=device)
    
#     # Use DataLoader to batch process
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=16, shuffle=False, num_workers=4
#     )
    
#     print("Calculating class weights...")
    
#     # Process in batches
#     for batch_idx, (_, masks) in enumerate(tqdm(dataloader)):
#         # Move masks to GPU
#         masks = masks.to(device)
        
#         # Count classes in batch (vectorized operation)
#         for class_idx in range(NUM_CLASSES):
#             class_counts[class_idx] += (masks == class_idx).sum().item()
        
#         # Report progress occasionally
#         if batch_idx % 50 == 0:
#             print(f"Processed {batch_idx * dataloader.batch_size}/{len(dataset)} images")
    
#     # Calculate weights (all on GPU)
#     epsilon = 1e-5
#     class_counts = class_counts + epsilon
    
#     # Calculate weights (inverse of frequency)
#     total_pixels = class_counts.sum()
#     class_weights = total_pixels / (class_counts * NUM_CLASSES)
    
#     # Normalize weights
#     class_weights = class_weights / class_weights.sum()
    
#     # Cap weights to avoid extremely high values
#     max_weight = 10.0
#     class_weights = torch.clamp(class_weights, max=max_weight)
    
#     print(f"Class counts: {class_counts}")
#     print(f"Class weights: {class_weights}")
    
#     return class_weights
CLASS_WEIGHTS_FILE = "class_weights.pth"  # File to store class weights


def calculate_class_weights(dataset):
    """
    Calculate and save class weights based on class frequencies with optimized GPU acceleration.
    If class weights exist, load them instead of recomputing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the class weights file exists
    if os.path.exists(CLASS_WEIGHTS_FILE):
        print("Loading precomputed class weights...")
        class_weights = torch.load(CLASS_WEIGHTS_FILE, map_location=device)
        return class_weights

    print("Class weights not found. Computing class weights...")

    # Compute class weights if not found
    batch_size = 64
    class_counts = torch.zeros(NUM_CLASSES, device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=False
    )

    print("Calculating class weights using GPU acceleration...")

    with torch.amp.autocast("cuda"):  # Use mixed precision for faster processing
        for _, masks in tqdm(dataloader):
            masks = masks.to(device)
            for class_idx in range(NUM_CLASSES):
                class_mask = (masks == class_idx).to(device)
                class_counts[class_idx] += class_mask.sum()
                del class_mask  # Free memory
                torch.cuda.empty_cache()

    epsilon = 1e-5
    class_counts += epsilon
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts * NUM_CLASSES)
    class_weights /= class_weights.sum()
    class_weights = torch.clamp(class_weights, max=10.0)

    print(f"Computed Class Weights: {class_weights}")

    # Save class weights to file
    torch.save(class_weights, CLASS_WEIGHTS_FILE)
    print(f"Class weights saved to {CLASS_WEIGHTS_FILE}")

    return class_weights