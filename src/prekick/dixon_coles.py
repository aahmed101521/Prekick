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
