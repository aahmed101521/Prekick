import pytest

from prekick.dixon_coles import dixon_coles_correction


def test_dixon_coles_correction_for_zero_zero():
    correction = dixon_coles_correction(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=0,
        away_goals=0,
        rho=-0.1,
    )

    assert correction == pytest.approx(1.12)


def test_dixon_coles_correction_for_zero_one():
    correction = dixon_coles_correction(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=0,
        away_goals=1,
        rho=-0.1,
    )

    assert correction == pytest.approx(0.85)


def test_dixon_coles_correction_for_one_zero():
    correction = dixon_coles_correction(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=1,
        away_goals=0,
        rho=-0.1,
    )

    assert correction == pytest.approx(0.92)


def test_dixon_coles_correction_for_one_one():
    correction = dixon_coles_correction(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=1,
        away_goals=1,
        rho=-0.1,
    )

    assert correction == pytest.approx(1.1)


def test_dixon_coles_correction_is_one_for_other_scores():
    correction = dixon_coles_correction(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=2,
        away_goals=1,
        rho=-0.1,
    )

    assert correction == 1.0
