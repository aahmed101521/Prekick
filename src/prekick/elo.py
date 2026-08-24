import math

from scipy.optimize import minimize_scalar


def expected_score(
    team_rating,
    opponent_rating,
):
    return 1 / (
        1
        + 10
        ** (
            (opponent_rating - team_rating)
            / 400
        )
    )


def three_way_probabilities(
    home_rating,
    away_rating,
    draw_parameter,
    home_advantage=100,
):
    if draw_parameter < 0:
        raise ValueError(
            "Draw parameter must be non-negative."
        )

    rating_difference = (
        home_rating
        + home_advantage
        - away_rating
    )

    strength_ratio = 10 ** (
        rating_difference / 400
    )

    draw_strength = (
        draw_parameter
        * strength_ratio ** 0.5
    )

    total_strength = (
        strength_ratio
        + draw_strength
        + 1
    )

    p_home = (
        strength_ratio
        / total_strength
    )

    p_draw = (
        draw_strength
        / total_strength
    )

    p_away = (
        1
        / total_strength
    )

    return p_home, p_draw, p_away


def update_rating(
    rating,
    actual_score,
    expected_score_value,
    k_factor=20,
):
    return rating + k_factor * (
        actual_score - expected_score_value
    )


def update_match(
    home_rating,
    away_rating,
    result,
    home_advantage=100,
    k_factor=20,
):
    expected_home = expected_score(
        home_rating + home_advantage,
        away_rating,
    )

    expected_away = 1 - expected_home

    if result == "H":
        actual_home = 1.0
        actual_away = 0.0
    elif result == "D":
        actual_home = 0.5
        actual_away = 0.5
    elif result == "A":
        actual_home = 0.0
        actual_away = 1.0
    else:
        raise ValueError(
            "Result must be 'H', 'D', or 'A'."
        )

    new_home_rating = update_rating(
        home_rating,
        actual_home,
        expected_home,
        k_factor,
    )

    new_away_rating = update_rating(
        away_rating,
        actual_away,
        expected_away,
        k_factor,
    )

    return new_home_rating, new_away_rating


def fit_draw_parameter(
    home_ratings,
    away_ratings,
    results,
    home_advantage=100,
):
    home_ratings = list(home_ratings)
    away_ratings = list(away_ratings)
    results = list(results)

    if not (
        len(home_ratings)
        == len(away_ratings)
        == len(results)
    ):
        raise ValueError(
            "Ratings and results must have the same length."
        )

    if len(results) == 0:
        raise ValueError(
            "At least one match is required."
        )

    def negative_log_likelihood(
        draw_parameter,
    ):
        total = 0.0

        for (
            home_rating,
            away_rating,
            result,
        ) in zip(
            home_ratings,
            away_ratings,
            results,
        ):
            p_home, p_draw, p_away = (
                three_way_probabilities(
                    home_rating,
                    away_rating,
                    draw_parameter,
                    home_advantage=home_advantage,
                )
            )

            if result == "H":
                actual_probability = p_home
            elif result == "D":
                actual_probability = p_draw
            elif result == "A":
                actual_probability = p_away
            else:
                raise ValueError(
                    "Result must be 'H', 'D', or 'A'."
                )

            total -= math.log(
                actual_probability
            )

        return total

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(1e-6, 10.0),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            "Draw parameter optimization failed."
        )

    return result.x
