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