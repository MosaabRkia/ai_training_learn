#config
# ✅ Training Configuration
trainingFromScratch = True
selfFixOptimization = True
useMaskForTraining = True

# ✅ GPU & Precision Settings
useMultiGPU = False
precisionMode = "fp16"

# ✅ Model & Training Parameters
MODEL_NAME = "models/checkpoint_epoch50.pth"
BEST_MODEL_PATH = "models/best_model.pth"
LOG_DIR = "logs/"

# ✅ Batch Size & Learning Rate
BATCH_SIZE_MIN = 2
BATCH_SIZE_MAX = 12
DEFAULT_BATCH_SIZE = 5
DEFAULT_LEARNING_RATE = 0.0001

# ✅ Image Resolution
TRAIN_IMAGE_WIDTH = 768
TRAIN_IMAGE_HEIGHT = 1024

# ✅ Automatic Optimization & Recovery
autoRecover = True
maxCheckpointHistory = 5
recoveryThreshold = 3

# ✅ Testing Parameters
numTestImagesDuringTraining = 3
TEST_IMAGE_DIR = "data/test/"
TEST_MODEL_DIR = "testModelsDuringTraining/"

