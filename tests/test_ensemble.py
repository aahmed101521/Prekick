import pytest

from prekick.ensemble import blend_probabilities


def test_blend_probabilities_equal_weight():
    blended = blend_probabilities(
        first_probabilities=(
            0.50,
            0.30,
            0.20,
        ),
        second_probabilities=(
            0.40,
            0.20,
            0.40,
        ),
        first_weight=0.5,
    )

    assert blended == pytest.approx(
        (
            0.45,
            0.25,
            0.30,
        )
    )


def test_blend_probabilities_rejects_invalid_weight():
    with pytest.raises(ValueError):
        blend_probabilities(
            first_probabilities=(
                0.50,
                0.30,
                0.20,
            ),
            second_probabilities=(
                0.40,
                0.20,
                0.40,
            ),
            first_weight=1.1,
        )
