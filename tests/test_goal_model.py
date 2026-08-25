import pytest
import numpy as np

from prekick.goal_model import (
    build_team_index,
    expected_goals,
    fit_goal_model,
    match_negative_log_likelihood,
    model_negative_log_likelihood,
    total_negative_log_likelihood,
    unpack_parameters,
    get_team_parameters,
)


def test_expected_goals():
    home_expected_goals, away_expected_goals = expected_goals(
        home_attack=0.3,
        home_defence=-0.1,
        away_attack=0.1,
        away_defence=0.2,
        home_advantage=0.25,
    )

    assert home_expected_goals == pytest.approx(2.117000016612675)
    assert away_expected_goals == pytest.approx(1.0)


def test_match_negative_log_likelihood():
    loss = match_negative_log_likelihood(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=2,
        away_goals=1,
    )

    assert loss == pytest.approx(2.4053605156578266)


def test_total_negative_log_likelihood():
    matches = [
        (1.5, 0.8, 2, 1),
        (1.2, 1.1, 1, 1),
    ]

    loss = total_negative_log_likelihood(matches)

    assert loss == pytest.approx(4.427728779059548)


def test_build_team_index():
    team_index = build_team_index(
        ["Chelsea", "Arsenal", "Liverpool"]
    )

    assert team_index == {
        "Arsenal": 0,
        "Chelsea": 1,
        "Liverpool": 2,
    }


def test_unpack_parameters():
    attacks, defences, home_advantage = unpack_parameters(
        [0.2, -0.1, 0.3, 0.0, -0.2, 0.25],
        number_of_teams=3,
    )

    assert attacks == pytest.approx([0.2, -0.1, -0.1])
    assert defences == pytest.approx([0.3, 0.0, -0.2])
    assert home_advantage == pytest.approx(0.25)
    assert sum(attacks) == pytest.approx(0.0)


def test_model_negative_log_likelihood():
    teams = [
        "Arsenal",
        "Chelsea",
        "Liverpool",
    ]

    team_index = build_team_index(teams)

    parameters = [
        0.2,
        -0.1,
        0.3,
        0.0,
        -0.2,
        0.25,
    ]

    matches = [
        ("Arsenal", "Chelsea", 2, 1),
        ("Liverpool", "Arsenal", 1, 1),
    ]

    loss = model_negative_log_likelihood(
        parameters,
        matches,
        team_index,
    )

    assert loss == pytest.approx(4.501174309700453)


def test_unpack_parameters_accepts_numpy_array():
    parameters = np.array(
        [0.2, -0.1, 0.3, 0.0, -0.2, 0.25]
    )

    attacks, defences, home_advantage = unpack_parameters(
        parameters,
        number_of_teams=3,
    )

    assert attacks == pytest.approx([0.2, -0.1, -0.1])
    assert defences == pytest.approx([0.3, 0.0, -0.2])
    assert home_advantage == pytest.approx(0.25)


def test_fit_goal_model():
    teams = [
        "Arsenal",
        "Chelsea",
        "Liverpool",
    ]

    matches = [
        ("Arsenal", "Chelsea", 2, 1),
        ("Chelsea", "Arsenal", 1, 1),
        ("Arsenal", "Liverpool", 1, 0),
        ("Liverpool", "Arsenal", 1, 1),
        ("Chelsea", "Liverpool", 1, 2),
        ("Liverpool", "Chelsea", 2, 1),
    ]

    attacks, defences, home_advantage = fit_goal_model(
        matches,
        teams,
    )

    assert len(attacks) == 3
    assert len(defences) == 3
    assert sum(attacks) == pytest.approx(0.0)
    assert home_advantage == pytest.approx(
        0.2876823801001614,
        abs=1e-5,
    )


def test_get_team_parameters_returns_fitted_values_for_known_team():
    team_index = {
        "Arsenal": 0,
        "Chelsea": 1,
    }
    attacks = [0.2, -0.2]
    defences = [0.1, 0.3]

    attack, defence = get_team_parameters(
        "Arsenal",
        team_index,
        attacks,
        defences,
    )

    assert attack == 0.2
    assert defence == 0.1


def test_get_team_parameters_uses_neutral_values_for_unseen_team():
    team_index = {
        "Arsenal": 0,
        "Chelsea": 1,
    }
    attacks = [0.2, -0.2]
    defences = [0.1, 0.3]

    attack, defence = get_team_parameters(
        "Fulham",
        team_index,
        attacks,
        defences,
    )

    assert attack == 0.0
    assert defence == 0.2
