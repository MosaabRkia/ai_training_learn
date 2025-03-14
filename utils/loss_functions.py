import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """ Custom Dice Loss for segmentation """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1e-5
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1 - dice

loss_fn = DiceLoss()
