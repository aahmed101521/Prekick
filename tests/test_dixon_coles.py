import pytest

from prekick.dixon_coles import (
    dixon_coles_correction,
    dixon_coles_match_negative_log_likelihood,
    dixon_coles_match_outcome_probabilities,
    dixon_coles_model_negative_log_likelihood,
    dixon_coles_score_probability,
    fit_rho,
    total_dixon_coles_negative_log_likelihood,
    unpack_dixon_coles_parameters,
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


def test_dixon_coles_match_negative_log_likelihood():
    loss = dixon_coles_match_negative_log_likelihood(
        home_expected_goals=1.5,
        away_expected_goals=0.8,
        home_goals=0,
        away_goals=0,
        rho=-0.1,
    )

    assert loss > 0


def test_dixon_coles_match_negative_log_likelihood_rejects_invalid_probability():
    with pytest.raises(ValueError):
        dixon_coles_match_negative_log_likelihood(
            home_expected_goals=10.0,
            away_expected_goals=10.0,
            home_goals=0,
            away_goals=0,
            rho=0.1,
        )


def test_total_dixon_coles_negative_log_likelihood():
    matches = [
        (
            1.5,
            0.8,
            0,
            0,
        ),
        (
            1.2,
            1.0,
            1,
            1,
        ),
    ]

    total_loss = total_dixon_coles_negative_log_likelihood(
        matches,
        rho=-0.1,
    )

    expected_loss = (
        dixon_coles_match_negative_log_likelihood(
            home_expected_goals=1.5,
            away_expected_goals=0.8,
            home_goals=0,
            away_goals=0,
            rho=-0.1,
        )
        + dixon_coles_match_negative_log_likelihood(
            home_expected_goals=1.2,
            away_expected_goals=1.0,
            home_goals=1,
            away_goals=1,
            rho=-0.1,
        )
    )

    assert total_loss == pytest.approx(
        expected_loss
    )


def test_fit_rho_recovers_known_low_score_pattern():
    matches = (
        [(1.0, 1.0, 0, 0)] * 6
        + [(1.0, 1.0, 1, 1)] * 5
        + [(1.0, 1.0, 0, 1)] * 5
        + [(1.0, 1.0, 1, 0)] * 4
    )

    rho = fit_rho(matches)

    assert rho == pytest.approx(
        -0.1,
        abs=1e-3,
    )


def test_unpack_dixon_coles_parameters():
    parameters = [
        0.2,
        -0.1,
        0.3,
        -0.2,
        0.1,
        0.15,
        -0.05,
    ]

    attacks, defences, home_advantage, rho = (
        unpack_dixon_coles_parameters(
            parameters,
            number_of_teams=3,
        )
    )

    assert attacks == pytest.approx(
        [
            0.2,
            -0.1,
            -0.1,
        ]
    )

    assert defences == pytest.approx(
        [
            0.3,
            -0.2,
            0.1,
        ]
    )

    assert home_advantage == pytest.approx(0.15)
    assert rho == pytest.approx(-0.05)


def test_dixon_coles_model_negative_log_likelihood():
    team_index = {
        "Team A": 0,
        "Team B": 1,
    }

    parameters = [
        0.2,
        0.1,
        -0.1,
        0.15,
        -0.05,
    ]

    matches = [
        (
            "Team A",
            "Team B",
            1,
            0,
        ),
    ]

    loss = dixon_coles_model_negative_log_likelihood(
        parameters,
        matches,
        team_index,
    )

    assert loss > 0
