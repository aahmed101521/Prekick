from math import exp, log

from scipy.optimize import minimize

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


def build_team_index(
    teams: list[str],
) -> dict[str, int]:
    """Return a mapping from team name to parameter index."""

    return {
        team: index
        for index, team in enumerate(sorted(teams))
    }


def get_team_parameters(
    team: str,
    team_index: dict[str, int],
    attacks: list[float],
    defences: list[float],
) -> tuple[float, float]:
    """Return attack and defence parameters for a team.

    Unseen teams receive neutral attack strength and the mean fitted defence.
    """

    if team in team_index:
        index = team_index[team]

        return (
            attacks[index],
            defences[index],
        )

    mean_defence = sum(defences) / len(defences)

    return 0.0, mean_defence


def unpack_parameters(
    parameters,
    number_of_teams: int,
) -> tuple[list[float], list[float], float]:
    """Split optimizer parameters into attack, defence, and home advantage."""

    parameter_values = list(parameters)

    expected_parameter_count = 2 * number_of_teams

    if len(parameter_values) != expected_parameter_count:
        raise ValueError(
            f"Expected {expected_parameter_count} parameters, "
            f"received {len(parameter_values)}."
        )

    fitted_attacks = parameter_values[
        : number_of_teams - 1
    ]

    final_attack = -sum(fitted_attacks)

    attacks = fitted_attacks + [final_attack]

    defence_start = number_of_teams - 1
    defence_end = defence_start + number_of_teams

    defences = parameter_values[
        defence_start:defence_end
    ]

    home_advantage = parameter_values[-1]

    return attacks, defences, home_advantage


def model_negative_log_likelihood(
    parameters,
    matches: list[tuple[str, str, int, int]],
    team_index: dict[str, int],
) -> float:
    """Return total model loss for observed matches."""

    number_of_teams = len(team_index)

    attacks, defences, home_advantage = unpack_parameters(
        parameters,
        number_of_teams,
    )

    total_loss = 0.0

    for (
        home_team,
        away_team,
        home_goals,
        away_goals,
    ) in matches:
        home_index = team_index[home_team]
        away_index = team_index[away_team]

        (
            home_expected_goals,
            away_expected_goals,
        ) = expected_goals(
            home_attack=attacks[home_index],
            home_defence=defences[home_index],
            away_attack=attacks[away_index],
            away_defence=defences[away_index],
            home_advantage=home_advantage,
        )

        total_loss += match_negative_log_likelihood(
            home_expected_goals=home_expected_goals,
            away_expected_goals=away_expected_goals,
            home_goals=home_goals,
            away_goals=away_goals,
        )

    return total_loss


def regularized_model_negative_log_likelihood(
    parameters,
    matches: list[tuple[str, str, int, int]],
    team_index: dict[str, int],
    penalty_strength: float,
) -> float:
    """Return model loss with L2 regularization."""

    likelihood_loss = model_negative_log_likelihood(
        parameters,
        matches,
        team_index,
    )

    attacks, defences, _ = unpack_parameters(
        parameters,
        len(team_index),
    )

    penalty = penalty_strength * (
        sum(
            value**2
            for value in attacks
        )
        + sum(
            value**2
            for value in defences
        )
    )

    return likelihood_loss + penalty


def fit_goal_model(
    matches: list[tuple[str, str, int, int]],
    teams: list[str],
):
    """Fit attack, defence, and home-advantage parameters."""

    team_index = build_team_index(teams)

    number_of_teams = len(team_index)

    initial_parameters = [
        0.0
    ] * (2 * number_of_teams)

    penalty_strength = 1.0

    result = minimize(
        regularized_model_negative_log_likelihood,
        initial_parameters,
        args=(
            matches,
            team_index,
            penalty_strength,
        ),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(
            f"Goal model optimization failed: {result.message}"
        )

    attacks, defences, home_advantage = unpack_parameters(
        result.x,
        number_of_teams,
    )

    return attacks, defences, home_advantage
