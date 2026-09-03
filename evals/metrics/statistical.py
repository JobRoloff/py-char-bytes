"""Statistical metrics for ground-truth alignment and model agreement evaluation."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def calculate_ground_truth_correlations(
    predictions: pd.Series | np.ndarray,
    ground_truth: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate Spearman rank (rho) and Pearson linear (r) correlations against ground truth.

    Args:
        predictions: Model predictions or readability metric scores.
        ground_truth: Educator-annotated scores (e.g., CLEAR BT_easiness).

    Returns:
        dict containing spearman_rho, spearman_pvalue, pearson_r, and pearson_pvalue.
    """
    # Clean non-numeric or missing entries
    valid_mask = pd.notna(predictions) & pd.notna(ground_truth)
    preds_clean = np.asarray(predictions)[valid_mask]
    gt_clean = np.asarray(ground_truth)[valid_mask]

    if len(preds_clean) < 3:
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "pearson_r": 0.0,
            "pearson_pvalue": 1.0,
        }

    rho_res = spearmanr(preds_clean, gt_clean)
    r_res = pearsonr(preds_clean, gt_clean)

    return {
        "spearman_rho": round(float(rho_res.statistic), 4),
        "spearman_pvalue": float(rho_res.pvalue),
        "pearson_r": round(float(r_res.statistic), 4),
        "pearson_pvalue": float(r_res.pvalue),
    }


def calculate_oracle_kappa(
    local_ratings: pd.Series | np.ndarray,
    oracle_ratings: pd.Series | np.ndarray,
    weights: str | None = "quadratic",
) -> float:
    """Calculate Cohen's Kappa for inter-model agreement (e.g., Gemma:2b vs Claude 3.5).

    Args:
        local_ratings: Ordinal clarity ratings from local SLM (1-10).
        oracle_ratings: Ordinal clarity ratings from Frontier Oracle model (1-10).
        weights: Weighting strategy ('linear', 'quadratic', or None). Default is 'quadratic'
            for ordinal scale distance penalization.

    Returns:
        Cohen's Kappa score bounded between -1.0 and 1.0.
    """
    valid_mask = pd.notna(local_ratings) & pd.notna(oracle_ratings)
    y1 = np.asarray(local_ratings)[valid_mask]
    y2 = np.asarray(oracle_ratings)[valid_mask]

    if len(y1) < 2:
        return 0.0

    kappa = cohen_kappa_score(y1, y2, weights=weights)
    return round(float(kappa), 4)