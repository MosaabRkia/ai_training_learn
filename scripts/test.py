import torch
import os
import cv2
from utils.model import load_model
from utils.data_loader import get_dataloader
from config import BEST_MODEL_PATH, TEST_IMAGE_DIR, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT

# ✅ Load Model
model = load_model(BEST_MODEL_PATH)
model.eval()

# ✅ Load Test Data
test_loader = get_dataloader(TEST_IMAGE_DIR, "", batch_size=1)

# ✅ Ensure Output Directory Exists
os.makedirs("testResults", exist_ok=True)

# ✅ Run Inference on Test Data
with torch.no_grad():
    for i, (image, _) in enumerate(test_loader):
        # ✅ Move image to GPU
        image = image.to("cuda")

        # ✅ Get Prediction
        output = model(image)

        # ✅ Convert Output to Mask
        predicted_mask = torch.argmax(output, dim=1).cpu().numpy()

        # ✅ Save the Output Mask
        output_path = f"testResults/result_{i}.png"
        cv2.imwrite(output_path, predicted_mask)
        print(f"✅ Saved Test Result: {output_path}")
