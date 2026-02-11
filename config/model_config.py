from pathlib import Path

TRAIN_TEST_SPLIT_DATE: str = "2024-07-01"
VALIDATION_FRACTION: float = 0.15

LIGHTGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

MAX_BOOST_ROUNDS: int = 2000
EARLY_STOPPING_ROUNDS: int = 50

TOP_N_FEATURES: int = 20

CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws", "platoon"]

MODELS_DIR = Path("data/models")
EVALUATION_DIR = MODELS_DIR / "evaluation"
MODEL_PATH = MODELS_DIR / "lgbm_hr_model.joblib"
