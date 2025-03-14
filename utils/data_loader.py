import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from config import useMaskForTraining, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT

# ✅ Define Augmentations
transform = A.Compose([
    A.Resize(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(p=0.1),
    A.CoarseDropout(max_holes=8, p=0.5), 
    ToTensorV2()
])

class GarmentDataset(Dataset):
    """ Dataset class for garment segmentation. """
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.use_mask = useMaskForTraining
        self.img_list = sorted(os.listdir(img_dir))
        self.mask_list = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH), dtype=np.uint8)

        augmented = transform(image=image, mask=mask)
        return augmented['image'].float() / 255.0, augmented['mask'].float()

# ✅ Load Data
def get_dataloader(img_dir, mask_dir, batch_size):
    dataset = GarmentDataset(img_dir, mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
