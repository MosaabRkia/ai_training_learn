import torch
import cv2
import os
import numpy as np
from utils.model import load_model
from config import BEST_MODEL_PATH, TEST_IMAGE_DIR, TEST_MODEL_DIR, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT

# ✅ Load Model
model = load_model(BEST_MODEL_PATH)
model.eval()

# ✅ Ensure Output Directory Exists
os.makedirs(TEST_MODEL_DIR, exist_ok=True)

# ✅ Process Test Images
for img_name in os.listdir(TEST_IMAGE_DIR):
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    # ✅ Resize image if needed
    image = cv2.resize(image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
    
    # ✅ Convert Image to Tensor
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0).to("cuda")

    # ✅ Run Inference
    with torch.no_grad():
        output = model(image)
    
    # ✅ Get the Prediction Mask
    predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    # ✅ Save the Parsed Image
    output_path = os.path.join(TEST_MODEL_DIR, f"parsed_{img_name}")
    cv2.imwrite(output_path, predicted_mask)
    print(f"✅ Saved Parsed Result: {output_path}")
