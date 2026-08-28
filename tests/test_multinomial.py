import numpy as np
import pytest

from prekick.multinomial import (
    linear_scores,
    multinomial_negative_log_likelihood,
    predict_probabilities,
    softmax,
    total_multinomial_negative_log_likelihood,
    unpack_multinomial_parameters,
)


def test_softmax_probabilities_sum_to_one():
    probabilities = softmax(
        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    assert probabilities.sum() == pytest.approx(1.0)


def test_softmax_probabilities_are_between_zero_and_one():
    probabilities = softmax(
        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    assert (
        (probabilities > 0)
        & (probabilities < 1)
    ).all()


def test_softmax_equal_scores_give_equal_probabilities():
    probabilities = softmax(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        )
    )

    assert probabilities == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )


def test_softmax_handles_large_scores():
    probabilities = softmax(
        np.array(
            [
                1000.0,
                1001.0,
                1002.0,
            ]
        )
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert probabilities.sum() == pytest.approx(1.0)


def test_linear_scores():
    features = np.array(
        [
            2.0,
            3.0,
        ]
    )

    coefficients = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    intercepts = np.array(
        [
            0.5,
            -0.5,
            1.0,
        ]
    )

    scores = linear_scores(
        features,
        coefficients,
        intercepts,
    )

    assert scores == pytest.approx(
        [
            2.5,
            2.5,
            6.0,
        ]
    )


def test_predict_probabilities():
    features = np.array(
        [
            2.0,
            3.0,
        ]
    )

    coefficients = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    intercepts = np.array(
        [
            0.5,
            -0.5,
            1.0,
        ]
    )

    probabilities = predict_probabilities(
        features,
        coefficients,
        intercepts,
    )

    assert probabilities.sum() == pytest.approx(1.0)

    assert (
        (probabilities > 0)
        & (probabilities < 1)
    ).all()

    assert probabilities[2] > probabilities[0]
    assert probabilities[2] > probabilities[1]


def test_multinomial_negative_log_likelihood():
    features = np.array(
        [
            1.0,
            2.0,
        ]
    )

    coefficients = np.zeros(
        (
            3,
            2,
        )
    )

    intercepts = np.zeros(3)

    loss = multinomial_negative_log_likelihood(
        features,
        coefficients,
        intercepts,
        outcome="H",
    )

    assert loss == pytest.approx(
        -np.log(1 / 3)
    )


def test_multinomial_negative_log_likelihood_rejects_invalid_outcome():
    features = np.array(
        [
            1.0,
            2.0,
        ]
    )

    coefficients = np.zeros(
        (
            3,
            2,
        )
    )

    intercepts = np.zeros(3)

    with pytest.raises(ValueError):
        multinomial_negative_log_likelihood(
            features,
            coefficients,
            intercepts,
            outcome="X",
        )


def test_total_multinomial_negative_log_likelihood():
    features = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )

    outcomes = [
        "H",
        "A",
    ]

    coefficients = np.zeros(
        (
            3,
            2,
        )
    )

    intercepts = np.zeros(3)

    total_loss = total_multinomial_negative_log_likelihood(
        features,
        outcomes,
        coefficients,
        intercepts,
    )

    expected_loss = (
        -np.log(1 / 3)
        - np.log(1 / 3)
    )

    assert total_loss == pytest.approx(
        expected_loss
    )


def test_unpack_multinomial_parameters():
    parameters = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
            -0.5,
        ]
    )

    coefficients, intercepts = (
        unpack_multinomial_parameters(
            parameters,
            number_of_features=2,
        )
    )

    assert np.array_equal(
        coefficients,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ]
        ),
    )

    assert np.array_equal(
        intercepts,
        np.array(
            [
                0.5,
                -0.5,
                0.0,
            ]
        ),
    )
