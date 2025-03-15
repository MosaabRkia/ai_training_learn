import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES

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


def calculate_class_weights(dataset):
    """
    Calculate class weights based on class frequencies with safeguards
    """
    class_counts = torch.zeros(NUM_CLASSES)
    
    # Count occurrences of each class
    for i, (_, mask) in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(dataset)} for class weights...")
        for class_idx in range(NUM_CLASSES):
            class_counts[class_idx] += (mask == class_idx).sum().item()
    
    # Add a small epsilon to prevent division by zero for classes that may not appear
    epsilon = 1e-5
    class_counts = class_counts + epsilon
    
    # Calculate weights (inverse of frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts * NUM_CLASSES)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    # Optional: Cap weights to avoid extremely high values
    max_weight = 10.0
    class_weights = torch.clamp(class_weights, max=max_weight)
    
    # Move to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    
    return class_weights