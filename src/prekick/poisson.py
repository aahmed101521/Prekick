from math import exp, factorial


def poisson_goal_probability(expected_goals: float, goals: int) -> float:
    """Return the probability of scoring exactly `goals` goals."""

    if expected_goals < 0:
        raise ValueError("expected_goals must be non-negative")

    if goals < 0:
        raise ValueError("goals must be non-negative")

    return (
        exp(-expected_goals)
        * expected_goals**goals
        / factorial(goals)
    )


def independent_score_probability(
    home_expected_goals: float,
    away_expected_goals: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """Return the independent Poisson probability of a match scoreline."""

    home_probability = poisson_goal_probability(
        home_expected_goals,
        home_goals,
    )
    away_probability = poisson_goal_probability(
        away_expected_goals,
        away_goals,
    )

    return home_probability * away_probability
