import numpy as np
import pytest

from prekick.multinomial import softmax


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
