import numpy as np

from prekick.preprocessing import (
    build_standardization_mask,
    fit_standardization_parameters,
    transform_features,
)


def test_fit_standardization_parameters():
    features = np.array(
        [
            [1.0, 10.0, 0.0],
            [3.0, 20.0, 1.0],
            [5.0, 30.0, 0.0],
        ]
    )

    standardize_mask = np.array(
        [
            True,
            True,
            False,
        ]
    )

    means, standard_deviations = (
        fit_standardization_parameters(
            features,
            standardize_mask,
        )
    )

    assert np.allclose(
        means,
        np.array(
            [
                3.0,
                20.0,
                0.0,
            ]
        ),
    )

    assert np.allclose(
        standard_deviations,
        np.array(
            [
                np.std([1.0, 3.0, 5.0]),
                np.std([10.0, 20.0, 30.0]),
                1.0,
            ]
        ),
    )


def test_transform_features():
    features = np.array(
        [
            [1.0, 10.0, 0.0],
            [np.nan, 20.0, 1.0],
            [5.0, 30.0, 0.0],
        ]
    )

    means = np.array(
        [
            3.0,
            20.0,
            0.0,
        ]
    )

    standard_deviations = np.array(
        [
            2.0,
            10.0,
            1.0,
        ]
    )

    transformed = transform_features(
        features,
        means,
        standard_deviations,
    )

    expected = np.array(
        [
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    assert np.allclose(
        transformed,
        expected,
    )


def test_build_standardization_mask():
    feature_names = [
        "home_goals_for_lag_1",
        "home_shots_for_lag_2",
        "home_has_lag_1_history",
        "away_goals_against_lag_1",
        "away_has_lag_2_history",
    ]

    mask = build_standardization_mask(
        feature_names
    )

    assert np.array_equal(
        mask,
        np.array(
            [
                True,
                True,
                False,
                True,
                False,
            ]
        ),
    )
