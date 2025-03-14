import torch
import torch.optim as optim
import os
import shutil
from torch.utils.data import DataLoader
from utils.model import load_model  # ✅ Ensures model is correctly imported
from utils.data_loader import get_dataloader  # ✅ Fixes dataset loader import
from utils.loss_functions import DiceLoss  # ✅ Ensures loss function is imported
from torch.utils.tensorboard import SummaryWriter
from config import *  # ✅ Imports all configurations

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # ✅ Forces Python to find utils

# ✅ Start Training
print("🚀 Training Script Started...")

# ✅ Training Settings
batch_size = DEFAULT_BATCH_SIZE
learning_rate = DEFAULT_LEARNING_RATE
best_loss = float("inf")
no_improve_epochs = 0
checkpoint_history = []  # Stores last `maxCheckpointHistory` models

# ✅ Load Data
print("📂 Loading Training Data...")
train_loader = get_dataloader("data/train/images/", "data/train/masks/", batch_size)
print(f"✅ Loaded {len(train_loader.dataset)} Training Images")

print("📂 Loading Validation Data...")
val_loader = get_dataloader("data/val/images/", "data/val/masks/", batch_size)
print(f"✅ Loaded {len(val_loader.dataset)} Validation Images")

# ✅ Load Model (New or Pre-Trained)
if trainingFromScratch:
    print("🚀 Starting Training from Scratch...")
    model = load_model()
elif os.path.exists(MODEL_NAME):
    print(f"✅ Loading Model from {MODEL_NAME}")
    model = load_model(MODEL_NAME)
else:
    print("⚠️ No Model Found! Starting Training from Scratch...")
    model = load_model()

# ✅ Enable Multi-GPU if Configured
if useMultiGPU and torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# ✅ Move Model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"⚡ Model Loaded on {device.upper()}")

# ✅ Optimizer & Loss Function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_fn = DiceLoss()

print("🎯 Starting Training Loop...")

# ✅ Training Loop
for epoch in range(100):
    print(f"\n🚀 Epoch [{epoch+1}/100] Started...")
    
    model.train()
    total_train_loss = 0

    print("📦 Processing Training Batches...")
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"   🔄 Batch {batch_idx + 1}/{len(train_loader)} Processing...")
        images, masks = images.to(device).float() / 255.0, masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"✅ Training Loss: {avg_train_loss:.4f}")

    # ✅ Validate Model
    model.eval()
    total_val_loss = 0
    print("📊 Validating Model...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device).float() / 255.0, masks.to(device).float()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"🔍 Validation Loss: {avg_val_loss:.4f}")

    # ✅ Save Best Model & Track Checkpoints
    checkpoint_path = f"models_checkpoints/checkpoint_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_history.append(checkpoint_path)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improve_epochs = 0  # Reset Counter
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🔥 Best Model Saved at Epoch {epoch+1}")
    else:
        no_improve_epochs += 1

    # ✅ Automatic Model Recovery (If Enabled)
    if autoRecover and no_improve_epochs >= recoveryThreshold:
        print("🚨 Performance Declining! Rolling Back to Previous Model...")

        if len(checkpoint_history) > 1:
            last_good_model = checkpoint_history[-2]  # Get second last model
            print(f"🔄 Reverting to {last_good_model}")
            model.load_state_dict(torch.load(last_good_model))

            # ✅ Reduce Learning Rate to Stabilize
            if learning_rate > LEARNING_RATE_MIN:
                learning_rate /= 2
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                print(f"🛠 Lowering Learning Rate to {learning_rate}")

            # ✅ Adjust Batch Size if Needed
            if batch_size > BATCH_SIZE_MIN:
                batch_size -= 2  # Reduce batch size for stability
                print(f"🔧 Adjusting Batch Size to {batch_size}")

            # ✅ Reload DataLoader with New Batch Size
            train_loader = get_dataloader("data/train/images/", "data/train/masks/", batch_size)
            val_loader = get_dataloader("data/val/images/", "data/val/masks/", batch_size)

            no_improve_epochs = 0  # Reset Counter

    # ✅ Test Model on Sample Images (If Enabled)
    if numTestImagesDuringTraining > 0:
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)
        os.system(f"python scripts/infer.py --model {BEST_MODEL_PATH} --input {TEST_IMAGE_DIR} --output {TEST_MODEL_DIR}")

    # ✅ Maintain Only `maxCheckpointHistory` Models
    if len(checkpoint_history) > maxCheckpointHistory:
        old_model = checkpoint_history.pop(0)  # Remove oldest checkpoint
        if os.path.exists(old_model):
            os.remove(old_model)
            print(f"🗑 Deleted Old Checkpoint: {old_model}")

print("\n✅ Training Completed Successfully! 🎉")
