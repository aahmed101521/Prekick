import pytest

from prekick.scoring import (
    brier_score,
    log_loss,
    ranked_probability_score,
    score_prediction,
)


def test_brier_score_home_win():
    score = brier_score(
        0.60,
        0.25,
        0.15,
        "H",
    )

    assert score == pytest.approx(0.245)


def test_brier_score_draw():
    score = brier_score(
        0.20,
        0.50,
        0.30,
        "D",
    )

    assert score == pytest.approx(0.38)


def test_brier_score_away_win():
    score = brier_score(
        0.10,
        0.20,
        0.70,
        "A",
    )

    assert score == pytest.approx(0.14)


def test_brier_score_invalid_result():
    with pytest.raises(
        ValueError,
        match="Result must be 'H', 'D', or 'A'.",
    ):
        brier_score(
            0.40,
            0.30,
            0.30,
            "X",
        )


def test_log_loss_home_win():
    score = log_loss(
        0.60,
        0.25,
        0.15,
        "H",
    )

    assert score == pytest.approx(
        0.5108256237659907
    )


def test_log_loss_draw():
    score = log_loss(
        0.20,
        0.50,
        0.30,
        "D",
    )

    assert score == pytest.approx(
        0.6931471805599453
    )


def test_log_loss_away_win():
    score = log_loss(
        0.10,
        0.20,
        0.70,
        "A",
    )

    assert score == pytest.approx(
        0.35667494393873245
    )


def test_log_loss_invalid_result():
    with pytest.raises(
        ValueError,
        match="Result must be 'H', 'D', or 'A'.",
    ):
        log_loss(
            0.40,
            0.30,
            0.30,
            "X",
        )


def test_log_loss_zero_probability():
    with pytest.raises(
        ValueError,
        match="Probability of the actual result must be greater than 0.",
    ):
        log_loss(
            0.00,
            0.50,
            0.50,
            "H",
        )


def test_ranked_probability_score_home_win():
    score = ranked_probability_score(
        0.60,
        0.25,
        0.15,
        "H",
    )

    assert score == pytest.approx(
        0.09125
    )


def test_ranked_probability_score_draw():
    score = ranked_probability_score(
        0.20,
        0.50,
        0.30,
        "D",
    )

    assert score == pytest.approx(
        0.065
    )


def test_ranked_probability_score_away_win():
    score = ranked_probability_score(
        0.10,
        0.20,
        0.70,
        "A",
    )

    assert score == pytest.approx(
        0.05
    )


def test_ranked_probability_score_invalid_result():
    with pytest.raises(
        ValueError,
        match="Result must be 'H', 'D', or 'A'.",
    ):
        ranked_probability_score(
            0.40,
            0.30,
            0.30,
            "X",
        )


def test_probabilities_must_be_between_zero_and_one():
    with pytest.raises(
        ValueError,
        match="Probabilities must be between 0 and 1.",
    ):
        brier_score(
            0.80,
            0.40,
            -0.20,
            "H",
        )


def test_probabilities_must_sum_to_one():
    with pytest.raises(
        ValueError,
        match="Probabilities must sum to 1.",
    ):
        brier_score(
            0.50,
            0.30,
            0.10,
            "H",
        )


def test_score_prediction_returns_all_metrics():
    scores = score_prediction(
        0.60,
        0.25,
        0.15,
        "H",
    )

    assert scores["rps"] == pytest.approx(
        0.09125
    )
    assert scores["log_loss"] == pytest.approx(
        0.5108256237659907
    )
    assert scores["brier"] == pytest.approx(
        0.245
    )