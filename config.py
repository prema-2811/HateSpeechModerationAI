# Dataset paths
TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

# Number of classes
NUM_LABELS = 3

# Label names
LABEL_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

# Final model
MODEL_NAME = "distilroberta-base"

# Training settings
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
RANDOM_SEED = 42