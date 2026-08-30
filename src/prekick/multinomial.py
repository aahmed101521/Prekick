import numpy as np

from scipy.optimize import minimize


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


def unpack_multinomial_parameters(
    parameters: np.ndarray,
    number_of_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert optimizer parameters into H/D/A coefficients and intercepts."""

    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    expected_length = (
        2 * number_of_features
        + 2
    )

    if len(parameters) != expected_length:
        raise ValueError(
            f"Expected {expected_length} parameters, "
            f"received {len(parameters)}."
        )

    coefficient_end = (
        2 * number_of_features
    )

    fitted_coefficients = parameters[
        :coefficient_end
    ].reshape(
        2,
        number_of_features,
    )

    fitted_intercepts = parameters[
        coefficient_end:
    ]

    coefficients = np.vstack(
        [
            fitted_coefficients,
            np.zeros(number_of_features),
        ]
    )

    intercepts = np.concatenate(
        [
            fitted_intercepts,
            np.zeros(1),
        ]
    )

    return coefficients, intercepts


def multinomial_model_negative_log_likelihood(
    parameters: np.ndarray,
    features: np.ndarray,
    outcomes: list[str],
) -> float:
    """Return multinomial model loss for optimizer parameters."""

    features = np.asarray(
        features,
        dtype=float,
    )

    number_of_features = features.shape[1]

    coefficients, intercepts = (
        unpack_multinomial_parameters(
            parameters,
            number_of_features,
        )
    )

    return total_multinomial_negative_log_likelihood(
        features,
        outcomes,
        coefficients,
        intercepts,
    )


def regularized_multinomial_model_negative_log_likelihood(
    parameters: np.ndarray,
    features: np.ndarray,
    outcomes: list[str],
    penalty_strength: float,
) -> float:
    """Return multinomial loss with L2 coefficient regularization."""

    features = np.asarray(
        features,
        dtype=float,
    )

    number_of_features = features.shape[1]

    base_loss = multinomial_model_negative_log_likelihood(
        parameters,
        features,
        outcomes,
    )

    coefficient_end = (
        2 * number_of_features
    )

    coefficient_parameters = parameters[
        :coefficient_end
    ]

    penalty = penalty_strength * np.sum(
        coefficient_parameters ** 2
    )

    return float(
        base_loss + penalty
    )


def fit_multinomial_model(
    features: np.ndarray,
    outcomes: list[str],
    penalty_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized multinomial logistic regression model."""

    features = np.asarray(
        features,
        dtype=float,
    )

    number_of_features = features.shape[1]

    number_of_parameters = (
        2 * number_of_features
        + 2
    )

    initial_parameters = np.zeros(
        number_of_parameters
    )

    result = minimize(
        regularized_multinomial_model_negative_log_likelihood,
        initial_parameters,
        args=(
            features,
            outcomes,
            penalty_strength,
        ),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(
            f"Multinomial model fitting failed: {result.message}"
        )

    return unpack_multinomial_parameters(
        result.x,
        number_of_features,
    )
