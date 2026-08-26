from math import log

from scipy.optimize import minimize_scalar

from prekick.goal_model import unpack_parameters
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


def dixon_coles_match_negative_log_likelihood(
    home_expected_goals: float,
    away_expected_goals: float,
    home_goals: int,
    away_goals: int,
    rho: float,
) -> float:
    """Return the Dixon-Coles negative log-likelihood of a scoreline."""

    probability = dixon_coles_score_probability(
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
        rho,
    )

    if probability <= 0:
        raise ValueError(
            "Dixon-Coles scoreline probability must be positive."
        )

    return -log(probability)


def total_dixon_coles_negative_log_likelihood(
    matches: list[tuple[float, float, int, int]],
    rho: float,
) -> float:
    """Return total Dixon-Coles negative log-likelihood across matches."""

    total_loss = 0.0

    for (
        home_expected_goals,
        away_expected_goals,
        home_goals,
        away_goals,
    ) in matches:
        total_loss += dixon_coles_match_negative_log_likelihood(
            home_expected_goals=home_expected_goals,
            away_expected_goals=away_expected_goals,
            home_goals=home_goals,
            away_goals=away_goals,
            rho=rho,
        )

    return total_loss


def unpack_dixon_coles_parameters(
    parameters,
    number_of_teams: int,
) -> tuple[list[float], list[float], float, float]:
    """Unpack goal-model parameters and Dixon-Coles rho."""

    goal_parameters = parameters[:-1]
    rho = float(parameters[-1])

    attacks, defences, home_advantage = unpack_parameters(
        goal_parameters,
        number_of_teams,
    )

    return (
        attacks,
        defences,
        home_advantage,
        rho,
    )


def fit_rho(
    matches: list[tuple[float, float, int, int]],
    lower_bound: float = -0.2,
    upper_bound: float = 0.2,
) -> float:
    """Fit the Dixon-Coles rho parameter for fixed expected goals."""

    def objective(rho: float) -> float:
        try:
            return total_dixon_coles_negative_log_likelihood(
                matches,
                rho,
            )
        except ValueError:
            return float("inf")

    result = minimize_scalar(
        objective,
        bounds=(
            lower_bound,
            upper_bound,
        ),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            f"Dixon-Coles rho optimization failed: {result.message}"
        )

    return float(result.x)


def dixon_coles_match_outcome_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Return Dixon-Coles home, draw, and away probabilities."""

    home_probability = 0.0
    draw_probability = 0.0
    away_probability = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            score_probability = dixon_coles_score_probability(
                home_expected_goals,
                away_expected_goals,
                home_goals,
                away_goals,
                rho,
            )

            if home_goals > away_goals:
                home_probability += score_probability
            elif home_goals == away_goals:
                draw_probability += score_probability
            else:
                away_probability += score_probability

    total_probability = (
        home_probability
        + draw_probability
        + away_probability
    )

    return (
        home_probability / total_probability,
        draw_probability / total_probability,
        away_probability / total_probability,
    )
