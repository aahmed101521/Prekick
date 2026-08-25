import pytest

from prekick.poisson import poisson_goal_probability


def test_poisson_goal_probability():
    probability = poisson_goal_probability(1.5, 2)

    assert probability == pytest.approx(0.25102143016698353)
