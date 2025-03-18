# ✅ Training Configuration
trainingFromScratch = False
selfFixOptimization = True
useMaskForTraining = True
DefineAugmentations = False  # Set to True to enable data augmentation

# ✅ GPU & Precision Settings
useMultiGPU = False
precisionMode = "fp32"  # Change to "fp32" from "fp16" to avoid precision issues

# ✅ Model & Training Parameters
MODEL_NAME = "models/best_model.pth"
BEST_MODEL_PATH = "models/best_model.pth"
LOG_DIR = "logs/"
NUM_CLASSES = 6  # 5 garment part classes + background (class 0)

# ✅ Class Labels & RGB Color Mapping
CLASS_MAPPING = {
    # Label: RGB values
    0: (0, 0, 0),       # Background
    1: (22, 22, 22),    # Left Arm
    2: (21, 21, 21),    # Right Arm
    3: (5, 5, 5),       # Chest/Middle
    4: (24, 24, 24),    # Collar (Front)
    5: (25, 25, 25),    # Body Back Parts
}

# ✅ Directory Structure
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
TRAIN_OUTPUT_DIR = "data/train/outputs/"
VAL_IMG_DIR = "data/val/images/"
VAL_MASK_DIR = "data/val/masks/"
VAL_OUTPUT_DIR = "data/val/outputs/"
TEST_IMAGE_DIR = "data/test/"
TEST_OUTPUT_DIR = "testResults/"

# ✅ Training Parameters
MAX_EPOCHS = 100
LEARNING_RATE_MIN = 1e-6
EARLY_STOPPING_PATIENCE = 10

# ✅ Batch Size & Learning Rate
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0001

# # ✅ Image Resolution
TRAIN_IMAGE_WIDTH = 768  # Half of original 768
TRAIN_IMAGE_HEIGHT = 1024  # Half of original 1024

# ✅ Automatic Optimization & Recovery
autoRecover = True
maxCheckpointHistory = 5
recoveryThreshold = 3

# ✅ Testing Parameters
numTestImagesDuringTraining = 3
TEST_MODEL_DIR = "testModelsDuringTraining/"