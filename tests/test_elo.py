import pytest

from prekick.elo import (
    expected_score,
    update_match,
    update_rating,
)


def test_expected_score_with_100_point_advantage():
    score = expected_score(
        1600,
        1500,
    )

    assert score == pytest.approx(
        0.6400649998028851
    )


def test_update_rating_after_home_win():
    expected = expected_score(
        1600,
        1500,
    )

    new_rating = update_rating(
        1500,
        1.0,
        expected,
    )

    assert new_rating == pytest.approx(
        1507.1987000039423
    )


def test_update_match_home_win():
    home_rating, away_rating = update_match(
        1500,
        1500,
        "H",
    )

    assert home_rating == pytest.approx(
        1507.1987000039423
    )
    assert away_rating == pytest.approx(
        1492.8012999960577
    )


def test_update_match_draw():
    home_rating, away_rating = update_match(
        1500,
        1500,
        "D",
    )

    assert home_rating == pytest.approx(
        1497.1987000039423
    )
    assert away_rating == pytest.approx(
        1502.8012999960577
    )


def test_update_match_invalid_result():
    with pytest.raises(
        ValueError,
        match="Result must be 'H', 'D', or 'A'.",
    ):
        update_match(
            1500,
            1500,
            "X",
        )


def test_three_way_probabilities_sum_to_one():
    from prekick.elo import three_way_probabilities

    probabilities = three_way_probabilities(
        1500,
        1500,
        draw_parameter=1.0,
    )

    assert abs(
        sum(probabilities) - 1.0
    ) < 1e-12


def test_three_way_probabilities_are_valid():
    from prekick.elo import three_way_probabilities

    probabilities = three_way_probabilities(
        1600,
        1500,
        draw_parameter=1.0,
    )

    assert all(
        0 <= probability <= 1
        for probability in probabilities
    )


def test_zero_draw_parameter_gives_zero_draw_probability():
    from prekick.elo import three_way_probabilities

    _, p_draw, _ = three_way_probabilities(
        1500,
        1500,
        draw_parameter=0.0,
    )

    assert p_draw == 0.0


def test_negative_draw_parameter_is_rejected():
    import pytest

    from prekick.elo import three_way_probabilities

    with pytest.raises(ValueError):
        three_way_probabilities(
            1500,
            1500,
            draw_parameter=-1.0,
        )


def test_stronger_home_team_has_higher_home_probability():
    from prekick.elo import three_way_probabilities

    p_home_equal, _, _ = three_way_probabilities(
        1500,
        1500,
        draw_parameter=1.0,
    )

    p_home_stronger, _, _ = three_way_probabilities(
        1600,
        1500,
        draw_parameter=1.0,
    )

    assert p_home_stronger > p_home_equal
