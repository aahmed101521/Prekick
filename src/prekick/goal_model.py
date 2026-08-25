from math import exp


def expected_goals(
    home_attack: float,
    home_defence: float,
    away_attack: float,
    away_defence: float,
    home_advantage: float,
) -> tuple[float, float]:
    """Return expected home and away goals."""

    home_expected_goals = exp(
        home_attack
        + away_defence
        + home_advantage
    )

    away_expected_goals = exp(
        away_attack
        + home_defence
    )

    return home_expected_goals, away_expected_goals
