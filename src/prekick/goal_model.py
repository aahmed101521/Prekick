from math import exp, log

from prekick.poisson import independent_score_probability


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


def match_negative_log_likelihood(
    home_expected_goals: float,
    away_expected_goals: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """Return the negative log-likelihood of an observed scoreline."""

    probability = independent_score_probability(
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
    )

    return -log(probability)


def total_negative_log_likelihood(
    matches: list[tuple[float, float, int, int]],
) -> float:
    """Return the total negative log-likelihood across multiple matches."""

    total_loss = 0.0

    for (
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
    ) in matches:
        total_loss += match_negative_log_likelihood(
            home_expected_goals=home_expected_goals,
            away_expected_goals=away_expected_goals,
            home_goals=home_goals,
            away_goals=away_goals,
        )

    return total_loss
