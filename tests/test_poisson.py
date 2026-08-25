import pytest

from prekick.poisson import (
    independent_score_probability,
    match_outcome_probabilities,
    poisson_goal_probability,
)


def test_poisson_goal_probability():
    probability = poisson_goal_probability(1.5, 2)

    assert probability == pytest.approx(0.25102143016698353)


def test_independent_score_probability():
    probability = independent_score_probability(1.5, 0.8, 2, 1)

    assert probability == pytest.approx(0.09023295935052335)


def test_match_outcome_probabilities():
    home_probability, draw_probability, away_probability = (
        match_outcome_probabilities(1.5, 0.8)
    )

    assert home_probability == pytest.approx(0.5381650462435573)
    assert draw_probability == pytest.approx(0.26185391289739474)
    assert away_probability == pytest.approx(0.19998048807020596)

    assert (
        home_probability
        + draw_probability
        + away_probability
    ) == pytest.approx(1.0, abs=1e-6)
