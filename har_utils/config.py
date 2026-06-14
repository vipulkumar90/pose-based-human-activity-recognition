from pathlib import Path

BASE_DIR = Path("data")
GRID_SEARCH_DIR = Path("results/grid_search")

BASE_FILE_NAME = "keypoints_with_labels"
FILE_TYPE = ".csv"
FILE_NAME_SUFFIX = ["1", "2", "3", "5"]

SHOULDER_WIDTH_THRESHOLD = 10

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': 2
}

HIST_GRADIENT_PARAMS = {
    'learning_rate': 0.1,
    'max_iter': 100,
    'max_depth': None,
    'min_samples_leaf': 20,
    'random_state': 42
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": 42
}

KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "minkowski"
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss"
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "random_state": 42
}

DECISION_TREE_PARAMS = {
    "max_depth": None,
    "random_state": 42
}

EXTRA_TREES_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1
}

ADABOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.5,
    "random_state": 42
}

GAUSSIAN_NB_PARAMS = {}

KMEANS_PARAMS = {
    "n_clusters": 14,
    "random_state": 42,
    "n_init": "auto"
}

DBSCAN_PARAMS = {
    "eps": 0.5,
    "min_samples": 5
}

AGGLOMERATIVE_PARAMS = {
    "n_clusters": 14
}