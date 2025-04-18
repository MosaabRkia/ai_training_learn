# infer.py
import torch
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.model import load_model
from utils.visualization import save_colored_mask, visualize_binary_mask
from utils.data_loader import mask_to_rgb
from config import CLASS_MAPPING, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, NUM_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", type=str, default="models/best_model.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="data/test/", 
                        help="Input directory containing images")
    parser.add_argument("--output", type=str, default="testResults/", 
                        help="Output directory for results")
    parser.add_argument("--architecture", type=str, default="unet", 
                        help="Model architecture (default: unet)")
    parser.add_argument("--encoder", type=str, default="efficientnet-b2", 
                        help="Encoder backbone (default: efficientnet-b2)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize predictions with overlay")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for binary segmentation (0-1)")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print(f"🔄 Loading model from {args.model}...")
    model = load_model(
        model_path=args.model,
        architecture=args.architecture,
        encoder=args.encoder
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"⚠️ No image files found in {args.input}")
        return
    
    print(f"🧪 Processing {len(image_files)} images...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(args.input, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Failed to read image: {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        
        # Resize the image for model input
        resized_image = cv2.resize(image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
        input_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # For binary segmentation, handle differently to get better visualization
        if NUM_CLASSES == 2:
            # Get probability map for hand class
            if output.shape[1] > 1:  # If output has multiple channels
                hand_prob = torch.softmax(output, dim=1)[:, 1]  # Get prob for class 1 (hand)
            else:
                hand_prob = torch.sigmoid(output[:, 0])  # Single channel case
                
            # Apply threshold
            hand_prob_np = hand_prob.cpu().numpy()[0]
            pred_mask = (hand_prob_np > args.confidence_threshold).astype(np.uint8)
            
            # Resize back to original size
            pred_mask_resized = cv2.resize(
                pred_mask, 
                (orig_width, orig_height), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            # Multi-class case - use argmax
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Resize back to original size
            pred_mask_resized = cv2.resize(
                pred_mask.astype(np.uint8), 
                (orig_width, orig_height), 
                interpolation=cv2.INTER_NEAREST
            )
        
        base_name = os.path.splitext(img_name)[0]
        
        # Save raw mask
        index_mask_path = os.path.join(args.output, f"{base_name}_mask.png")
        cv2.imwrite(index_mask_path, pred_mask_resized)
        
        # Save colored mask
        colored_mask_path = os.path.join(args.output, f"{base_name}_colored.png")
        save_colored_mask(pred_mask_resized, colored_mask_path, CLASS_MAPPING)
        
        # Create and save visualization
        if args.visualize:
            if NUM_CLASSES == 2:
                # Use specialized hand visualization
                blended = visualize_binary_mask(image, pred_mask_resized, title="Hand Segmentation", alpha=0.5)
            else:
                # Standard overlay for multi-class
                colored_mask = mask_to_rgb(pred_mask_resized, CLASS_MAPPING)
                blended = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                
            blended_path = os.path.join(args.output, f"{base_name}_overlay.png")
            cv2.imwrite(blended_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            
            # For binary segmentation, also save a version with confidence map
            if NUM_CLASSES == 2:
                # Resize probability map to original size
                hand_prob_resized = cv2.resize(hand_prob_np, (orig_width, orig_height))
                
                # Create heatmap visualization
                heatmap = cv2.applyColorMap((hand_prob_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Blend with original image
                heatmap_blend = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
                heatmap_path = os.path.join(args.output, f"{base_name}_heatmap.png")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_blend, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Processing complete! Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)