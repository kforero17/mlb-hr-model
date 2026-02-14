import logging

import numpy as np
import pandas as pd

from src.models.starter_k_evaluation import predict_over_under

logger = logging.getLogger(__name__)


def predict_lambda(
    model: object,
    features: pd.DataFrame,
) -> np.ndarray:
    return model.predict(features)


def predict_game(
    model: object,
    features: pd.DataFrame,
    lines: list[float],
    dispersion: float,
) -> pd.DataFrame:
    lambda_pred = predict_lambda(model, features)

    result = pd.DataFrame({"lambda_pred": lambda_pred})

    for line in lines:
        p_over = predict_over_under(lambda_pred, line, dispersion)
        result[f"p_over_{line}"] = p_over

    logger.info(
        "Predicted %d games — mean lambda: %.2f | lines: %s",
        len(lambda_pred), float(np.mean(lambda_pred)),
        ", ".join(f"{l}" for l in lines),
    )

    return result
