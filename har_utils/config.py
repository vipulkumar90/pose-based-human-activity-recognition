from pathlib import Path

BASE_DIR = Path("data")
BASE_FILE_NAME = "keypoints_with_labels"
FILE_TYPE = ".csv"
FILE_NAME_SUFFIX = ["1", "2", "3", "5"]

SHOULDER_WIDTH_THRESHOLD = 10

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

HIST_GRADIENT_PARAMS = {
    'learning_rate': 0.1,
    'max_iter': 100,
    'max_depth': None,
    'min_samples_leaf': 20,
    'random_state': 42
}
