import pytest

from prekick.goal_model import (
    expected_goals,
    match_negative_log_likelihood,
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
