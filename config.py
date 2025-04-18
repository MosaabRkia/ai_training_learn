# config.py - Optimized for Hand Segmentation

# ✅ Training Configuration
trainingFromScratch = True
selfFixOptimization = True
useMaskForTraining = True
DefineAugmentations = True  # Enable data augmentation for better generalization

# ✅ GPU & Precision Settings
useMultiGPU = True
precisionMode = "fp32"  # Use full precision for better stability with binary segmentation

# ✅ Model & Training Parameters
MODEL_NAME = "models/checkpoint_epoch4.pth"
BEST_MODEL_PATH = "models/best_model.pth"
LOG_DIR = "logs/hand_segmentation/"

# ✅ Class Labels & RGB Color Mapping
CLASS_MAPPING = {
    # Label: RGB values
    0: (0, 0, 0),       # Background
    1: (255, 255, 255)  # Hands
}

NUM_CLASSES = len(CLASS_MAPPING)  # 2 classes (background + hands)

# ✅ Directory Structure
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
TRAIN_OUTPUT_DIR = "data/train/outputs/"
VAL_IMG_DIR = "data/val/images/"
VAL_MASK_DIR = "data/val/masks/"
VAL_OUTPUT_DIR = "data/val/outputs/"
TEST_IMAGE_DIR = "data/test/"
TEST_OUTPUT_DIR = "testResults/"

# ✅ Training Parameters - Adjusted for Binary Segmentation
MAX_EPOCHS = 100
LEARNING_RATE_MIN = 1e-6
EARLY_STOPPING_PATIENCE = 15  # Increased patience for binary segmentation

# ✅ Batch Size & Learning Rate - Adjusted for Binary Segmentation
BATCH_SIZE_MIN = 8
BATCH_SIZE_MAX = 24
DEFAULT_BATCH_SIZE = 16        # Reduced batch size for better gradients
DEFAULT_LEARNING_RATE = 5e-5   # Lower learning rate for finer details

# ✅ Image Resolution - Increased for Better Hand Details
TRAIN_IMAGE_WIDTH = 512   # Increased from 384
TRAIN_IMAGE_HEIGHT = 512  # Increased from 512

# ✅ Automatic Optimization & Recovery
autoRecover = True
maxCheckpointHistory = 5
recoveryThreshold = 5  # Increased for binary segmentation

# ✅ Testing Parameters
numTestImagesDuringTraining = 5  # Increased for better validation
TEST_MODEL_DIR = "testModelsDuringTraining/"

# ✅ Binary Segmentation Specific Settings
BINARY_SEGMENTATION = True     # Flag to enable binary-specific optimizations
CLASS_WEIGHTS_FILE = "binary_class_weights.pth"  # Separate file for binary weights