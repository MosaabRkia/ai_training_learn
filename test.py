# test.py
import torch
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.model import load_model
from utils.data_loader import get_val_dataloader, rgb_to_mask, rgb_to_binary_mask
from utils.visualization import visualize_prediction, create_confusion_visualization, visualize_hand_segmentation_metrics
from config import CLASS_MAPPING, NUM_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on validation or test dataset")
    parser.add_argument("--model", type=str, default="models/best_model.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--img-dir", type=str, default="data/test", 
                        help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="data/val/outputs", 
                        help="Directory containing ground truth outputs")
    parser.add_argument("--output", type=str, default="testResults/", 
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for testing")
    parser.add_argument("--architecture", type=str, default="unet", 
                        help="Model architecture")
    parser.add_argument("--encoder", type=str, default="efficientnet-b2", 
                        help="Encoder backbone")
    return parser.parse_args()

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print(f"🔄 Loading model from {args.model}...")
    model = load_model(
        model_path=args.model,
        architecture=args.architecture,
        encoder=args.encoder
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Load test data
    print(f"📂 Loading test data from {args.img_dir} and {args.output_dir}...")
    test_loader = get_val_dataloader(args.img_dir, args.output_dir, batch_size=args.batch_size)
    print(f"✅ Loaded {len(test_loader.dataset)} test images")
    
    # Initialize metrics
    class_intersection = np.zeros(NUM_CLASSES)
    class_union = np.zeros(NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    # Process each batch
    print("🧪 Running inference on test data...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader)):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Run inference
            outputs = model(images)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            for i in range(images.size(0)):
                pred = predictions[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # Update confusion matrix
                batch_confusion = confusion_matrix(
                    mask.flatten(), pred.flatten(), 
                    labels=list(range(NUM_CLASSES))
                )
                confusion += batch_confusion
                
                # Calculate IoU for each class
                for class_idx in range(NUM_CLASSES):
                    pred_mask = (pred == class_idx)
                    true_mask = (mask == class_idx)
                    
                    intersection = (pred_mask & true_mask).sum()
                    union = (pred_mask | true_mask).sum()
                    
                    class_intersection[class_idx] += intersection
                    class_union[class_idx] += union
                
                # Create visualizations - save more for binary segmentation to better evaluate hands
                if batch_idx < 20:  # Save visualizations for first 20 batches
                    img = images[i].cpu()
                    
                    # Basic visualization
                    vis = visualize_prediction(img, mask, pred)
                    vis_path = os.path.join(args.output, f"vis_batch{batch_idx}_sample{i}.png")
                    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    
                    # Confusion visualization
                    confusion_vis = create_confusion_visualization(img, mask, pred)
                    conf_path = os.path.join(args.output, f"conf_batch{batch_idx}_sample{i}.png")
                    cv2.imwrite(conf_path, cv2.cvtColor(confusion_vis, cv2.COLOR_RGB2BGR))
                    
                    # For binary segmentation, add specialized hand segmentation metrics visualization
                    if NUM_CLASSES == 2:
                        hand_metrics_vis = visualize_hand_segmentation_metrics(img, mask, pred)
                        metrics_vis_path = os.path.join(args.output, f"hand_metrics_batch{batch_idx}_sample{i}.png")
                        cv2.imwrite(metrics_vis_path, cv2.cvtColor(hand_metrics_vis, cv2.COLOR_RGB2BGR))
    
    # Calculate IoU for each class
    epsilon = 1e-10  # Small constant to avoid division by zero
    class_iou = class_intersection / (class_union + epsilon)
    
    # Calculate precision and recall from confusion matrix
    precision = np.zeros(NUM_CLASSES)
    recall = np.zeros(NUM_CLASSES)
    f1_score = np.zeros(NUM_CLASSES)
    
    for i in range(NUM_CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp + epsilon)
        recall[i] = tp / (tp + fn + epsilon)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + epsilon)
    
    # Print results
    print("\n📊 Test Results:")
    print(f"Mean IoU: {class_iou.mean():.4f}")
    
    # Define class names based on binary segmentation for hands
    if NUM_CLASSES == 2:
        class_names = {
            0: "Background",
            1: "Hand"
        }
    else:
        class_names = {
            0: "Background",
            1: "Left Arm",
            2: "Right Arm",
            3: "Chest/Middle",
            4: "Collar (Front)",
            5: "Body Back Parts"
        }
    
    print("\nClass-wise metrics:")
    print(f"{'Class':<15} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    for i in range(NUM_CLASSES):
        print(f"{class_names.get(i, f'Class {i}'):<15} {class_iou[i]:.4f}    {precision[i]:.4f}     {recall[i]:.4f}     {f1_score[i]:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mean IoU: {class_iou.mean():.4f}\n\n")
        f.write("Class-wise metrics:\n")
        f.write(f"{'Class':<15} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(NUM_CLASSES):
            f.write(f"{class_names.get(i, f'Class {i}'):<15} {class_iou[i]:.4f}    {precision[i]:.4f}     {recall[i]:.4f}     {f1_score[i]:.4f}\n")
    
    # For binary segmentation, calculate additional hand-specific metrics
    if NUM_CLASSES == 2:
        # Calculate hand detection rate (what percentage of images with hands are detected correctly)
        hand_detection_rate = 0
        images_with_hands = 0
        
        for i in range(NUM_CLASSES):
            row_sum = confusion[i, :].sum()
            if row_sum > 0:  # If there are any true samples for this class
                class_accuracy = confusion[i, i] / row_sum
                if i == 1:  # Hand class
                    hand_detection_rate = class_accuracy
                    images_with_hands = row_sum
        
        # Append hand-specific metrics to the file
        with open(metrics_path, "a") as f:
            f.write("\n\nHand Segmentation Specific Metrics:\n")
            f.write(f"Hand Detection Rate: {hand_detection_rate:.4f}\n")
            f.write(f"Total Images with Hands: {images_with_hands}\n")
            f.write(f"Hand IoU: {class_iou[1]:.4f}\n")
            f.write(f"Hand Precision: {precision[1]:.4f}\n")
            f.write(f"Hand Recall: {recall[1]:.4f}\n")
        
        print("\nHand Segmentation Specific Metrics:")
        print(f"Hand Detection Rate: {hand_detection_rate:.4f}")
        print(f"Total Images with Hands: {images_with_hands}")
    
    print(f"\n✅ Test complete! Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)