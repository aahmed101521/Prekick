import pytest

from prekick.goal_model import expected_goals


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
