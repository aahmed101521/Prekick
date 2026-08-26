from prekick.poisson import independent_score_probability


def dixon_coles_correction(
    home_expected_goals: float,
    away_expected_goals: float,
    home_goals: int,
    away_goals: int,
    rho: float,
) -> float:
    """Return the Dixon-Coles correction factor for a scoreline."""

    if home_goals == 0 and away_goals == 0:
        return (
            1
            - home_expected_goals
            * away_expected_goals
            * rho
        )

    if home_goals == 0 and away_goals == 1:
        return (
            1
            + home_expected_goals
            * rho
        )

    if home_goals == 1 and away_goals == 0:
        return (
            1
            + away_expected_goals
            * rho
        )

    if home_goals == 1 and away_goals == 1:
        return 1 - rho

    return 1.0


def dixon_coles_score_probability(
    home_expected_goals: float,
    away_expected_goals: float,
    home_goals: int,
    away_goals: int,
    rho: float,
) -> float:
    """Return the Dixon-Coles-adjusted probability of a scoreline."""

    independent_probability = independent_score_probability(
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
    )

    correction = dixon_coles_correction(
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
        rho,
    )

    return independent_probability * correction
