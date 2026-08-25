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


def match_outcome_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Return independent Poisson home, draw, and away probabilities."""

    home_probability = 0.0
    draw_probability = 0.0
    away_probability = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            score_probability = independent_score_probability(
                home_expected_goals,
                away_expected_goals,
                home_goals,
                away_goals,
            )

            if home_goals > away_goals:
                home_probability += score_probability
            elif home_goals == away_goals:
                draw_probability += score_probability
            else:
                away_probability += score_probability

    return home_probability, draw_probability, away_probability
