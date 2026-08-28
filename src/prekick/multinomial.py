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


def multinomial_negative_log_likelihood(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    outcome: str,
) -> float:
    """Return negative log-likelihood for one H/D/A outcome."""

    outcome_index = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    if outcome not in outcome_index:
        raise ValueError(
            "Outcome must be 'H', 'D', or 'A'."
        )

    probabilities = predict_probabilities(
        features,
        coefficients,
        intercepts,
    )

    observed_probability = probabilities[
        outcome_index[outcome]
    ]

    return float(
        -np.log(observed_probability)
    )


def total_multinomial_negative_log_likelihood(
    features: np.ndarray,
    outcomes: list[str],
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> float:
    """Return total multinomial negative log-likelihood."""

    features = np.asarray(
        features,
        dtype=float,
    )

    if len(features) != len(outcomes):
        raise ValueError(
            "Features and outcomes must contain the same number of rows."
        )

    total_loss = 0.0

    for match_features, outcome in zip(
        features,
        outcomes,
    ):
        total_loss += multinomial_negative_log_likelihood(
            match_features,
            coefficients,
            intercepts,
            outcome,
        )

    return total_loss
