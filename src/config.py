import os
from dotenv import load_dotenv
load_dotenv()
python_path = os.getenv("PYTHONPATH")
if python_path and python_path not in os.sys.path:
    os.sys.path.append(python_path)

print("PYTHONPATH:", python_path)

DATA_DIR = os.getenv("DATA_DIR", "./data")
TRAIN_PATH = os.getenv("TRAIN_PATH", f"{DATA_DIR}/train.csv")
TEST_PATH = os.getenv("TEST_PATH", f"{DATA_DIR}/test.csv")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
LR = float(os.getenv("LR", 0.001))

