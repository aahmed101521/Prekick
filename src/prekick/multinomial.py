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
