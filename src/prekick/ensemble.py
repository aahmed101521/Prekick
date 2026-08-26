def blend_probabilities(
    first_probabilities: tuple[float, float, float],
    second_probabilities: tuple[float, float, float],
    first_weight: float = 0.5,
) -> tuple[float, float, float]:
    """Return a weighted blend of two H/D/A probability forecasts."""

    if not 0 <= first_weight <= 1:
        raise ValueError(
            "Blend weight must be between 0 and 1."
        )

    second_weight = 1 - first_weight

    return tuple(
        first_weight * first_probability
        + second_weight * second_probability
        for first_probability, second_probability
        in zip(
            first_probabilities,
            second_probabilities,
        )
    )
