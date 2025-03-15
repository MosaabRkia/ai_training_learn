import torch
import torch.optim as optim
import os
import shutil
import time
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.model import load_model
from utils.data_loader import get_train_dataloader, get_val_dataloader
from utils.loss_functions import CombinedLoss, calculate_class_weights
from utils.visualization import visualize_prediction, save_colored_mask

# Import config
from config import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train garment segmentation model")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, 
                        help=f"Maximum epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, 
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--scratch", action="store_true", 
                        help="Train from scratch regardless of existing models")
    parser.add_argument("--architecture", type=str, default="deeplabv3plus", 
                        help="Model architecture (default: deeplabv3plus)")
    parser.add_argument("--encoder", type=str, default="resnet50", 
                        help="Encoder backbone (default: resnet50)")
    return parser.parse_args()

# ✅ Training Function
def train(args):
    print("🚀 Training Script Started...")
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models_checkpoints", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ✅ Training Settings
    batch_size = args.batch_size
    learning_rate = args.lr
    max_epochs = args.max_epochs
    best_loss = float("inf")
    no_improve_epochs = 0
    checkpoint_history = []  # Stores last `maxCheckpointHistory` models
    
    # ✅ Initialize TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    # ✅ Load Data
    print("📂 Loading Training Data...")
    train_loader = get_train_dataloader(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR if useMaskForTraining else None, 
        TRAIN_OUTPUT_DIR, 
        batch_size
    )
    print(f"✅ Loaded {len(train_loader.dataset)} Training Images")
    
    print("📂 Loading Validation Data...")
    val_loader = get_val_dataloader(
        VAL_IMG_DIR, 
        VAL_MASK_DIR if useMaskForTraining else None, 
        VAL_OUTPUT_DIR, 
        batch_size
    )
    print(f"✅ Loaded {len(val_loader.dataset)} Validation Images")
    
    # ✅ Load or Create Model
    if args.scratch or trainingFromScratch or not os.path.exists(MODEL_NAME):
        print("🚀 Starting Training from Scratch...")
        model = load_model(architecture=args.architecture, encoder=args.encoder)
    else:
        print(f"✅ Loading Model from {MODEL_NAME}")
        model = load_model(MODEL_NAME, architecture=args.architecture, encoder=args.encoder)
        
    # ✅ Enable Multi-GPU if Configured
    if useMultiGPU and torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    # ✅ Move Model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Convert to half precision after moving to GPU if specified
    if precisionMode == "fp16":
        model = model.half()
        
    print(f"⚡ Model Loaded on {device.upper()} with precision {precisionMode}")
    
    # ✅ Optimizer & Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ✅ Initialize Loss Function with Class Weights
    class_weights = calculate_class_weights(train_loader.dataset)
    loss_fn = CombinedLoss(dice_weight=0.7, focal_weight=0.3, class_weights=class_weights)
    
    print("🎯 Starting Training Loop...")
    
    # ✅ Training Loop
    for epoch in range(max_epochs):
        print(f"\n🚀 Epoch [{epoch+1}/{max_epochs}] Started...")
        
        # Set model to training mode
        model.train()
        total_train_loss = 0
        
        # Training progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for batch_idx, (images, masks) in progress_bar:
            # Ensure images are in the right format
            images = images.to(device)
            masks = masks.to(device)
            
            # Debug info
            # if batch_idx == 0:
            #     print(f"Image tensor shape: {images.shape}, dtype: {images.dtype}")
            #     print(f"Mask tensor shape: {masks.shape}, dtype: {masks.dtype}")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"✅ Training Loss: {avg_train_loss:.4f}")
        
        # Log training loss
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # ✅ Validate Model
        model.eval()
        total_val_loss = 0
        
        # Initialize metrics
        intersection_sum = 0
        union_sum = 0
        
        print("📊 Validating Model...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                total_val_loss += loss.item()
                
                # Calculate IoU metrics
                preds = torch.argmax(outputs, dim=1)
                for class_idx in range(1, NUM_CLASSES):  # Skip background class
                    pred_mask = (preds == class_idx)
                    true_mask = (masks == class_idx)
                    
                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    
                    if union > 0:
                        intersection_sum += intersection
                        union_sum += union
        
        # Calculate average validation loss and IoU
        avg_val_loss = total_val_loss / len(val_loader)
        mean_iou = intersection_sum / (union_sum + 1e-10)
        
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
        
        # Log validation metrics
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Metrics/mIoU', mean_iou, epoch)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # ✅ Save Checkpoint
        checkpoint_path = f"models_checkpoints/checkpoint_epoch{epoch+1}.pth"
        
        # Save model state
        model_state = {
            'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) 
                              else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_val_loss,
            'architecture': args.architecture,
            'encoder': args.encoder
        }
        
        torch.save(model_state, checkpoint_path)
        checkpoint_history.append(checkpoint_path)
        
        # ✅ Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0  # Reset Counter
            
            # Save best model
            torch.save(model_state, BEST_MODEL_PATH)
            print(f"🔥 Best Model Saved at Epoch {epoch+1} with Loss: {avg_val_loss:.4f}")
            
            # Visualize some predictions from best model
            if epoch > 0:  # Skip first epoch
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data
                    val_images, val_masks = next(iter(val_loader))
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)
                    
                    # Get predictions
                    val_outputs = model(val_images)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    
                    # Visualize first 4 images
                    for i in range(min(4, val_images.size(0))):
                        # Get image, mask, and prediction
                        img = val_images[i].cpu()
                        mask = val_masks[i].cpu()
                        pred = val_preds[i].cpu()
                        
                        # Visualize prediction
                        vis = visualize_prediction(img, mask, pred)
                        
                        # Add to TensorBoard
                        writer.add_image(f'predictions/sample_{i}', vis, epoch, dataformats='HWC')
        else:
            no_improve_epochs += 1
            print(f"⚠️ No Improvement for {no_improve_epochs} Epochs")
        
        # ✅ Early Stopping
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early Stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement")
            break
            
        # ✅ Automatic Model Recovery (If Enabled)
        if autoRecover and no_improve_epochs >= recoveryThreshold:
            print("🚨 Performance Declining! Rolling Back to Previous Model...")
            
            if len(checkpoint_history) > 1:
                last_good_model = checkpoint_history[-2]  # Get second last model
                print(f"🔄 Reverting to {last_good_model}")
                
                # Load previous model
                checkpoint = torch.load(last_good_model)
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                # ✅ Reduce Learning Rate to Stabilize
                if learning_rate > LEARNING_RATE_MIN:
                    learning_rate /= 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    print(f"🛠 Lowering Learning Rate to {learning_rate}")
                
                # ✅ Adjust Batch Size if Needed
                if batch_size > BATCH_SIZE_MIN:
                    batch_size = max(batch_size - 2, BATCH_SIZE_MIN)  # Reduce batch size for stability
                    print(f"🔧 Adjusting Batch Size to {batch_size}")
                    
                    # ✅ Reload DataLoader with New Batch Size
                    train_loader = get_train_dataloader(
                        TRAIN_IMG_DIR, 
                        TRAIN_MASK_DIR if useMaskForTraining else None,
                        TRAIN_OUTPUT_DIR,
                        batch_size
                    )
                    val_loader = get_val_dataloader(
                        VAL_IMG_DIR, 
                        VAL_MASK_DIR if useMaskForTraining else None,
                        VAL_OUTPUT_DIR,
                        batch_size
                    )
                
                no_improve_epochs = 0  # Reset Counter
        
        # ✅ Test Model on Sample Images (If Enabled)
        if numTestImagesDuringTraining > 0 and TEST_IMAGE_DIR:
            os.makedirs(TEST_MODEL_DIR, exist_ok=True)
            
            # Run inference script on test images
            print(f"🧪 Testing on {numTestImagesDuringTraining} sample images...")
            cmd = f"python infer.py --model {BEST_MODEL_PATH} --input {TEST_IMAGE_DIR} --output {TEST_MODEL_DIR} --architecture {args.architecture} --encoder {args.encoder}"
            os.system(cmd)
        
        # ✅ Maintain Only `maxCheckpointHistory` Models
        if len(checkpoint_history) > maxCheckpointHistory:
            old_model = checkpoint_history.pop(0)  # Remove oldest checkpoint
            if os.path.exists(old_model):
                os.remove(old_model)
                print(f"🗑 Deleted Old Checkpoint: {old_model}")
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\n✅ Training Completed in {training_time/3600:.2f} hours! 🎉")
    print(f"📊 Best Validation Loss: {best_loss:.4f}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)