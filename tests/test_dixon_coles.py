import pytest

from prekick.dixon_coles import (
    dixon_coles_correction,
    dixon_coles_match_outcome_probabilities,
    dixon_coles_score_probability,
)
from prekick.poisson import match_outcome_probabilities


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


def test_dixon_coles_score_probability_adjusts_low_score():
    probability = dixon_coles_score_probability(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=0,
        away_goals=0,
        rho=-0.1,
    )

    assert probability == pytest.approx(
        0.10025884372280375 * 1.12
    )


def test_dixon_coles_score_probability_leaves_other_score_unchanged():
    probability = dixon_coles_score_probability(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=2,
        away_goals=1,
        rho=-0.1,
    )

    assert probability == pytest.approx(
        0.09023295935052335
    )


def test_dixon_coles_match_probabilities_sum_to_one():
    probabilities = dixon_coles_match_outcome_probabilities(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        rho=-0.1,
    )

    assert sum(probabilities) == pytest.approx(1.0)


def test_zero_rho_matches_independent_poisson():
    dixon_coles_probabilities = (
        dixon_coles_match_outcome_probabilities(
            home_expected_goals=1.5,
            away_expected_goals=0.8,
            rho=0.0,
        )
    )

    independent_probabilities = match_outcome_probabilities(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
    )

    assert dixon_coles_probabilities == pytest.approx(
        independent_probabilities
    )


def test_negative_rho_increases_draw_probability():
    dixon_coles_probabilities = (
        dixon_coles_match_outcome_probabilities(
            home_expected_goals=1.5,
            away_expected_goals=0.8,
            rho=-0.1,
        )
    )

    independent_probabilities = match_outcome_probabilities(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
    )

    assert (
        dixon_coles_probabilities[1]
        > independent_probabilities[1]
    )
