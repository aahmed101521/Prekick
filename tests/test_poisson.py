import pytest

from prekick.poisson import (
    independent_score_probability,
    poisson_goal_probability,
)


def test_poisson_goal_probability():
    probability = poisson_goal_probability(1.5, 2)

    assert probability == pytest.approx(0.25102143016698353)


def test_independent_score_probability():
    probability = independent_score_probability(1.5, 0.8, 2, 1)

    assert probability == pytest.approx(0.09023295935052335)
