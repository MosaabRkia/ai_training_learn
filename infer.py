import torch
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.model import load_model
from utils.visualization import save_colored_mask
from utils.data_loader import mask_to_rgb
from config import CLASS_MAPPING, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", type=str, default="models/checkpoint_epoch4.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="data/test/", 
                        help="Input directory containing images")
    parser.add_argument("--output", type=str, default="testResults/", 
                        help="Output directory for results")
    parser.add_argument("--architecture", type=str, default="deeplabv3plus", 
                        help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet50", 
                        help="Encoder backbone")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize predictions with overlay")
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
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"⚠️ No image files found in {args.input}")
        return
    
    print(f"🧪 Processing {len(image_files)} images...")
    
    # Process each image
    for img_name in tqdm(image_files):
        img_path = os.path.join(args.input, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Failed to read image: {img_path}")
            continue
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original dimensions for later resizing
        orig_height, orig_width = image.shape[:2]
        
        # Resize image to model input size
        resized_image = cv2.resize(image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
        
        # Preprocess image
        input_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get prediction
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Resize prediction back to original size
        pred_mask_resized = cv2.resize(
            pred_mask.astype(np.uint8), 
            (orig_width, orig_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Get base filename without extension
        base_name = os.path.splitext(img_name)[0]
        
        # Save class index mask
        index_mask_path = os.path.join(args.output, f"{base_name}_mask.png")
        cv2.imwrite(index_mask_path, pred_mask_resized)
        
        # Save colored mask
        colored_mask_path = os.path.join(args.output, f"{base_name}_colored.png")
        save_colored_mask(pred_mask_resized, colored_mask_path, CLASS_MAPPING)
        
        # Create and save visualization if requested
        if args.visualize:
            # Convert RGB class mask
            colored_mask = mask_to_rgb(pred_mask_resized, CLASS_MAPPING)
            
            # Create blended image (70% original, 30% mask)
            alpha = 0.7
            beta = 0.3
            blended = cv2.addWeighted(
                image, alpha, 
                colored_mask, beta, 
                0
            )
            
            # Save blended image
            blended_path = os.path.join(args.output, f"{base_name}_overlay.png")
            cv2.imwrite(blended_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Processing complete! Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)