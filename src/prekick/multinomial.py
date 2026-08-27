import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert model scores into probabilities that sum to one."""

    scores = np.asarray(
        scores,
        dtype=float,
    )

    shifted_scores = scores - np.max(scores)

    exponentials = np.exp(
        shifted_scores
    )

    return exponentials / exponentials.sum()


def linear_scores(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Return linear Home, Draw, and Away class scores."""

    features = np.asarray(
        features,
        dtype=float,
    )

    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    intercepts = np.asarray(
        intercepts,
        dtype=float,
    )

    return coefficients @ features + intercepts


def predict_probabilities(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Return Home, Draw, and Away probabilities."""

    scores = linear_scores(
        features,
        coefficients,
        intercepts,
    )

    return softmax(scores)
