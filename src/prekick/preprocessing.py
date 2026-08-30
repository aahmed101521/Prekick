import numpy as np


def fit_standardization_parameters(
    features: np.ndarray,
    standardize_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn feature means and standard deviations from training data."""

    features = np.asarray(
        features,
        dtype=float,
    )

    standardize_mask = np.asarray(
        standardize_mask,
        dtype=bool,
    )

    if features.shape[1] != len(standardize_mask):
        raise ValueError(
            "Standardization mask must match the number of features."
        )

    means = np.zeros(
        features.shape[1],
        dtype=float,
    )

    standard_deviations = np.ones(
        features.shape[1],
        dtype=float,
    )

    means[standardize_mask] = np.nanmean(
        features[:, standardize_mask],
        axis=0,
    )

    standard_deviations[standardize_mask] = np.nanstd(
        features[:, standardize_mask],
        axis=0,
    )

    standard_deviations[
        standard_deviations == 0
    ] = 1.0

    return means, standard_deviations
