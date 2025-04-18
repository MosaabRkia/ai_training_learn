# model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import NUM_CLASSES, precisionMode
import os

class GarmentSegModel(nn.Module):
    """
    Garment Segmentation Model using segmentation_models_pytorch
    """
    def __init__(self, num_classes=NUM_CLASSES, architecture="deeplabv3plus", 
                 encoder="resnet50", encoder_weights="imagenet"):
        super(GarmentSegModel, self).__init__()
        
        # Available architectures
        architectures = {
            "unet": smp.Unet,
            "unet++": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
        }
        
        # Choose architecture
        model_fn = architectures.get(architecture.lower(), smp.DeepLabV3Plus)
        
        # Create model
        self.model = model_fn(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None  # No activation, we'll use softmax in loss
        )
        
    def forward(self, x):
        # Ensure input tensor is in the right format for the model
        # For FP16 training, ensure input is also half precision
        if precisionMode == "fp16" and x.dtype != torch.float16:
            x = x.half()
        return self.model(x)


class BinaryHandSegmentationModel(nn.Module):
    """
    Specialized model for binary hand segmentation with optimized settings
    """
    def __init__(self, num_classes=2):
        super(BinaryHandSegmentationModel, self).__init__()
        
        # Use UNet with efficient-net backbone for better performance on binary tasks
        self.model = smp.Unet(
            encoder_name="efficientnet-b2", 
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, x):
        # Ensure input tensor is in the right format for the model
        if precisionMode == "fp16" and x.dtype != torch.float16:
            x = x.half()
        return self.model(x)


class HighResolutionGarmentModel(nn.Module):
    """
    High Resolution Network (HRNet) for Garment Segmentation
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(HighResolutionGarmentModel, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-hrnet_w48",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        
    def forward(self, x):
        # Ensure input tensor is in the right format for the model
        if precisionMode == "fp16" and x.dtype != torch.float16:
            x = x.half()
        return self.model(x)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple segmentation models
    """
    def __init__(self, models_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        
    def forward(self, x):
        # Ensure input tensor is in the right format for the models
        if precisionMode == "fp16" and x.dtype != torch.float16:
            x = x.half()
            
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        
        # Average the predictions
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        
        return ensemble_output


class BinaryEnsembleModel(nn.Module):
    """
    Ensemble specifically designed for binary hand segmentation
    """
    def __init__(self, num_classes=2):
        super(BinaryEnsembleModel, self).__init__()
        
        # Create an ensemble of models optimized for binary segmentation
        self.models = nn.ModuleList([
            smp.Unet(
                encoder_name="efficientnet-b2",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None
            ),
            smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None
            ),
            smp.FPN(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        ])
    
    def forward(self, x):
        # Ensure input tensor is in the right format
        if precisionMode == "fp16" and x.dtype != torch.float16:
            x = x.half()
            
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        
        # Average the predictions
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        
        return ensemble_output


# ✅ Model Factory Function
def create_model(architecture="unet", encoder="efficientnet-b2", 
                 encoder_weights="imagenet", num_classes=NUM_CLASSES):
    """
    Factory function to create segmentation model - optimized for binary hand segmentation
    """
    # Special models
    if architecture.lower() == "hrnet":
        return HighResolutionGarmentModel(num_classes=num_classes)
    elif architecture.lower() == "ensemble":
        if num_classes == 2:
            # Use binary-specific ensemble for hand segmentation
            return BinaryEnsembleModel(num_classes=num_classes)
        else:
            # Use regular ensemble for multi-class segmentation
            models = [
                GarmentSegModel(num_classes=num_classes, architecture="deeplabv3plus", encoder="resnet50"),
                GarmentSegModel(num_classes=num_classes, architecture="unet++", encoder="efficientnet-b4"),
                GarmentSegModel(num_classes=num_classes, architecture="fpn", encoder="resnext101_32x8d")
            ]
            return EnsembleModel(models)
    elif architecture.lower() == "binary" and num_classes == 2:
        # Specialized binary segmentation model
        return BinaryHandSegmentationModel(num_classes=num_classes)
    else:
        # Regular segmentation model
        return GarmentSegModel(num_classes=num_classes, architecture=architecture, 
                               encoder=encoder, encoder_weights=encoder_weights)


# ✅ Load Model Function
def load_model(model_path=None, architecture="unet", 
               encoder="efficientnet-b2", num_classes=NUM_CLASSES):
    """
    Load a model from checkpoint or create a new one - optimized for hand segmentation
    """
    # Create model
    model = create_model(architecture, encoder, "imagenet", num_classes)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Handle DataParallel models
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            print(f"✅ Loaded model weights from: {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            print(f"⚠️ Creating new model instead...")
    
    # Set precision mode - but don't convert the model yet
    # (we'll do this after moving to GPU to avoid issues)
    
    return model