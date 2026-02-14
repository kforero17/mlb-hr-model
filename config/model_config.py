from pathlib import Path

HOLDOUT_DATE: str = "2024-07-01"
TRAIN_TEST_SPLIT_DATE: str = HOLDOUT_DATE
VALIDATION_FRACTION: float = 0.15

CV_VAL_WINDOW_DAYS: int = 45
CV_STEP_DAYS: int = 30
CV_MIN_TRAIN_DAYS: int = 365

LIGHTGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 200,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

MAX_BOOST_ROUNDS: int = 2000
EARLY_STOPPING_ROUNDS: int = 50

TUNING_N_TRIALS: int = 50
TUNING_CV_FOLDS: int = 3
TUNING_BRIER_WEIGHT: float = 2.0

TOP_N_FEATURES: int = 20

CALIBRATION_METHOD: str = "platt"

PREGAME_MODE: bool = True

CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws", "platoon", "inning", "roof_type"]
PREGAME_CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws", "platoon", "roof_type"]

MODELS_DIR = Path("data/models")
EVALUATION_DIR = MODELS_DIR / "evaluation"
MODEL_PATH = MODELS_DIR / "lgbm_k_model.joblib"
TUNED_PARAMS_PATH = MODELS_DIR / "tuned_params.json"

STARTER_K_LGBM_PARAMS: dict = {
    "objective": "poisson",
    "metric": "poisson",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

STARTER_K_MAX_BOOST_ROUNDS: int = 2000
STARTER_K_EARLY_STOPPING_ROUNDS: int = 50

STARTER_K_CATEGORICAL_FEATURES: list[str] = ["p_throws", "roof_type"]

STARTER_K_MODELS_DIR = MODELS_DIR / "starter_k"
STARTER_K_EVALUATION_DIR = STARTER_K_MODELS_DIR / "evaluation"
STARTER_K_MODEL_PATH = STARTER_K_MODELS_DIR / "lgbm_starter_k_model.joblib"

STARTER_K_OVER_UNDER_LINES: list[float] = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
