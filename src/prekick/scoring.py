import math


def _validate_probabilities(
    p_home,
    p_draw,
    p_away,
):
    probabilities = [
        p_home,
        p_draw,
        p_away,
    ]

    if any(
        probability < 0 or probability > 1
        for probability in probabilities
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not math.isclose(
        sum(probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Probabilities must sum to 1."
        )


def brier_score(
    p_home,
    p_draw,
    p_away,
    result,
):
    _validate_probabilities(
        p_home,
        p_draw,
        p_away,
    )

    if result == "H":
        actual = [1, 0, 0]
    elif result == "D":
        actual = [0, 1, 0]
    elif result == "A":
        actual = [0, 0, 1]
    else:
        raise ValueError(
            "Result must be 'H', 'D', or 'A'."
        )

    probabilities = [
        p_home,
        p_draw,
        p_away,
    ]

    score = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(
            probabilities,
            actual,
        )
    )

    return score


def log_loss(
    p_home,
    p_draw,
    p_away,
    result,
):
    _validate_probabilities(
        p_home,
        p_draw,
        p_away,
    )

    if result == "H":
        actual_probability = p_home
    elif result == "D":
        actual_probability = p_draw
    elif result == "A":
        actual_probability = p_away
    else:
        raise ValueError(
            "Result must be 'H', 'D', or 'A'."
        )

    if actual_probability <= 0:
        raise ValueError(
            "Probability of the actual result must be greater than 0."
        )

    return -math.log(actual_probability)


def ranked_probability_score(
    p_home,
    p_draw,
    p_away,
    result,
):
    _validate_probabilities(
        p_home,
        p_draw,
        p_away,
    )

    if result == "H":
        actual = [1, 0, 0]
    elif result == "D":
        actual = [0, 1, 0]
    elif result == "A":
        actual = [0, 0, 1]
    else:
        raise ValueError(
            "Result must be 'H', 'D', or 'A'."
        )

    probabilities = [
        p_home,
        p_draw,
        p_away,
    ]

    predicted_cumulative = [
        probabilities[0],
        probabilities[0] + probabilities[1],
    ]

    actual_cumulative = [
        actual[0],
        actual[0] + actual[1],
    ]

    score = sum(
        (predicted - observed) ** 2
        for predicted, observed in zip(
            predicted_cumulative,
            actual_cumulative,
        )
    ) / 2

    return score


def score_prediction(
    p_home,
    p_draw,
    p_away,
    result,
):
    return {
        "rps": ranked_probability_score(
            p_home,
            p_draw,
            p_away,
            result,
        ),
        "log_loss": log_loss(
            p_home,
            p_draw,
            p_away,
            result,
        ),
        "brier": brier_score(
            p_home,
            p_draw,
            p_away,
            result,
        ),
    }