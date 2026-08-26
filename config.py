from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

TARGET = "tuketim"
ID_COL = "id"
ENTITY_COL = "tanim"
DATE_COL = "tarih"

RANDOM_STATE = 42
VALID_DAYS = 60

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 1500,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

CATEGORICAL_FEATURES = [
    ENTITY_COL,
    "lokasyon",
    "il",
    "bolge",
    "ilce",
]

NUMERIC_FEATURES = [
    "guc",
    "guc_log",
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]
