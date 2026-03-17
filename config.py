import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "startupdata.csv")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "startup_success_final_model.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "startup_success_imputer.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2