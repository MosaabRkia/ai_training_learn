import torch
import segmentation_models_pytorch as smp

class GarmentSegModel(torch.nn.Module):
    def __init__(self, num_classes=5, pretrained_model="resnet50"):
        super(GarmentSegModel, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=pretrained_model,  # resnet50 or hrnetv2
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

# ✅ Load Model
def load_model(model_path=None):
    model = GarmentSegModel(num_classes=5, pretrained_model="resnet50") 
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded Model from: {model_path}")
    return model
