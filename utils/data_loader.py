import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import (
    useMaskForTraining, 
    TRAIN_IMAGE_WIDTH, 
    TRAIN_IMAGE_HEIGHT,
    CLASS_MAPPING,
    precisionMode
)

# ✅ Define RGB to Class Index Mapping Function
def rgb_to_mask(mask_rgb, class_mapping):
    """
    Convert RGB mask to single-channel class index mask
    
    Args:
        mask_rgb: RGB mask image (H, W, 3)
        class_mapping: Dictionary mapping class indices to RGB values
        
    Returns:
        mask: Single-channel class index mask (H, W)
    """
    height, width = mask_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert RGB to class indices
    for class_idx, rgb_value in class_mapping.items():
        # Create a binary mask for pixels matching this RGB value
        match_pixels = np.all(mask_rgb == rgb_value, axis=2)
        mask[match_pixels] = class_idx
        
    return mask

# ✅ Define Class Index to RGB Mapping Function
def mask_to_rgb(mask, class_mapping):
    """
    Convert single-channel class index mask to RGB mask
    
    Args:
        mask: Single-channel class index mask (H, W)
        class_mapping: Dictionary mapping class indices to RGB values
        
    Returns:
        mask_rgb: RGB mask image (H, W, 3)
    """
    height, width = mask.shape
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert class indices to RGB
    for class_idx, rgb_value in class_mapping.items():
        # Create a binary mask for pixels with this class index
        match_pixels = mask == class_idx
        mask_rgb[match_pixels] = rgb_value
        
    return mask_rgb

# ✅ Define Augmentations
train_transform = A.Compose([
    A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
    ToTensorV2()
])

class GarmentDataset(Dataset):
    """Dataset class for garment segmentation."""
    def __init__(self, img_dir, mask_dir=None, output_dir=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.transform = transform
        self.is_test = is_test
        self.use_mask = useMaskForTraining and mask_dir is not None
        
        # Get image list
        if os.path.exists(img_dir):
            self.img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.img_list = []
            print(f"Warning: Image directory {img_dir} does not exist!")
            
        # Get mask list if using masks and not in test mode
        if self.use_mask and not is_test and os.path.exists(mask_dir):
            self.mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Verify image-mask pairs
            if len(self.img_list) != len(self.mask_list):
                print(f"Warning: Number of images ({len(self.img_list)}) doesn't match masks ({len(self.mask_list)})!")
        else:
            self.mask_list = []
            
        # Get output segmentation list if not in test mode
        if not is_test and output_dir and os.path.exists(output_dir):
            self.output_list = sorted([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Verify image-output pairs
            if len(self.img_list) != len(self.output_list):
                print(f"Warning: Number of images ({len(self.img_list)}) doesn't match outputs ({len(self.output_list)})!")
        else:
            self.output_list = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # In test mode, return only the image
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                # Ensure proper data type for model input
                img_tensor = augmented['image'].float() / 255.0
                return img_tensor, self.img_list[idx]
            
            # Manual conversion without augmentation
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return img_tensor, self.img_list[idx]
        
        # Load mask/label - prioritize output segmentations if available
        if self.output_dir and len(self.output_list) > 0:
            output_path = os.path.join(self.output_dir, self.output_list[idx])
            mask_rgb = cv2.imread(output_path)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            mask = rgb_to_mask(mask_rgb, CLASS_MAPPING)
        elif self.use_mask:
            # Use mask as alternative ground truth
            mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
            mask_rgb = cv2.imread(mask_path)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            mask = rgb_to_mask(mask_rgb, CLASS_MAPPING)
        else:
            # Create empty mask if neither is available
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            # Ensure proper data type for model input
            img_tensor = augmented['image'].float() / 255.0
            return img_tensor, augmented['mask'].long()
        
        # Manual conversion without augmentation
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor

# ✅ Data Loader Functions
def get_train_dataloader(img_dir, mask_dir=None, output_dir=None, batch_size=4):
    """Get training data loader with augmentations"""
    dataset = GarmentDataset(img_dir, mask_dir, output_dir, transform=train_transform)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

def get_val_dataloader(img_dir, mask_dir=None, output_dir=None, batch_size=4):
    """Get validation data loader without augmentations"""
    dataset = GarmentDataset(img_dir, mask_dir, output_dir, transform=val_transform)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )

def get_test_dataloader(img_dir, batch_size=1):
    """Get test data loader without masks or outputs"""
    dataset = GarmentDataset(img_dir, is_test=True, transform=val_transform)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
    )