from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("predictions/backtest_scored.csv")
OUTPUT_PATH = Path("results/heldout_rps_block_bootstrap.csv")

EXPECTED_MATCHES = 1520
EXPECTED_BLOCKS = 460

ELO_WEIGHT_SELECTED = 0.25
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260903

REFERENCE_MODEL = "Elo + Poisson 50/50"

REQUIRED_COLUMNS = [
    "Date",
    "FTR",
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
    "ensemble_rps",
    "elo_rps",
    "poisson_rps",
    "dixon_coles_rps",
    "market_rps",
    "base_rate_rps",
]


def ranked_probability_score_vectorized(
    p_home,
    p_draw,
    results,
):
    p_home = np.asarray(p_home, dtype=float)
    p_draw = np.asarray(p_draw, dtype=float)
    results = np.asarray(results)

    first_actual = np.select(
        [results == "H", results == "D", results == "A"],
        [1.0, 0.0, 0.0],
        default=np.nan,
    )
    second_actual = np.select(
        [results == "H", results == "D", results == "A"],
        [1.0, 1.0, 0.0],
        default=np.nan,
    )

    if np.isnan(first_actual).any() or np.isnan(second_actual).any():
        raise ValueError("FTR contains a value other than H, D, or A.")

    predicted_first = p_home
    predicted_second = p_home + p_draw

    return (
        (predicted_first - first_actual) ** 2
        + (predicted_second - second_actual) ** 2
    ) / 2.0


def build_model_rps(backtest):
    selected_home = (
        ELO_WEIGHT_SELECTED * backtest["elo_home_prob"]
        + (1.0 - ELO_WEIGHT_SELECTED)
        * backtest["poisson_home_prob"]
    )
    selected_draw = (
        ELO_WEIGHT_SELECTED * backtest["elo_draw_prob"]
        + (1.0 - ELO_WEIGHT_SELECTED)
        * backtest["poisson_draw_prob"]
    )
    selected_away = (
        ELO_WEIGHT_SELECTED * backtest["elo_away_prob"]
        + (1.0 - ELO_WEIGHT_SELECTED)
        * backtest["poisson_away_prob"]
    )

    probability_sum = selected_home + selected_draw + selected_away
    if not np.allclose(
        probability_sum.to_numpy(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Validation-selected ensemble probabilities do not sum to 1."
        )

    selected_rps = ranked_probability_score_vectorized(
        selected_home,
        selected_draw,
        backtest["FTR"],
    )

    model_rps = pd.DataFrame(
        {
            "Date": backtest["Date"],
            "Elo + Poisson 50/50": backtest["ensemble_rps"],
            "Validation-selected Elo 25% / Poisson 75%": selected_rps,
            "Elo": backtest["elo_rps"],
            "Dixon-Coles": backtest["dixon_coles_rps"],
            "Independent Poisson": backtest["poisson_rps"],
            "Market benchmark": backtest["market_rps"],
            "Base rate": backtest["base_rate_rps"],
        }
    )

    if model_rps.isna().any().any():
        raise ValueError("Missing values found in model RPS data.")

    return model_rps


def aggregate_prediction_date_blocks(model_rps):
    model_columns = [
        column
        for column in model_rps.columns
        if column != "Date"
    ]

    block_sizes = (
        model_rps.groupby("Date", sort=True)
        .size()
        .to_numpy(dtype=float)
    )

    block_sums = (
        model_rps.groupby("Date", sort=True)[model_columns]
        .sum()
        .to_numpy(dtype=float)
    )

    return model_columns, block_sizes, block_sums


def paired_block_bootstrap(
    model_columns,
    block_sizes,
    block_sums,
):
    rng = np.random.default_rng(RANDOM_SEED)

    n_blocks = len(block_sizes)
    n_models = len(model_columns)

    reference_index = model_columns.index(REFERENCE_MODEL)

    comparator_indices = [
        index
        for index in range(n_models)
        if index != reference_index
    ]

    bootstrap_deltas = np.empty(
        (
            BOOTSTRAP_REPLICATES,
            len(comparator_indices),
        ),
        dtype=float,
    )

    # Work in batches to keep memory use modest.
    batch_size = 500
    completed = 0

    while completed < BOOTSTRAP_REPLICATES:
        current_batch = min(
            batch_size,
            BOOTSTRAP_REPLICATES - completed,
        )

        sampled_blocks = rng.integers(
            0,
            n_blocks,
            size=(current_batch, n_blocks),
        )

        sampled_sizes = block_sizes[sampled_blocks].sum(axis=1)

        sampled_sums = block_sums[sampled_blocks].sum(axis=1)

        sampled_means = (
            sampled_sums
            / sampled_sizes[:, np.newaxis]
        )

        reference_means = sampled_means[:, reference_index]

        for output_index, comparator_index in enumerate(
            comparator_indices
        ):
            bootstrap_deltas[
                completed : completed + current_batch,
                output_index,
            ] = (
                reference_means
                - sampled_means[:, comparator_index]
            )

        completed += current_batch

    return comparator_indices, bootstrap_deltas


def main():
    backtest = pd.read_csv(INPUT_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in backtest.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if len(backtest) != EXPECTED_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_MATCHES} held-out matches, "
            f"found {len(backtest)}."
        )

    block_count = backtest["Date"].nunique()
    if block_count != EXPECTED_BLOCKS:
        raise ValueError(
            f"Expected {EXPECTED_BLOCKS} prediction-date blocks, "
            f"found {block_count}."
        )

    model_rps = build_model_rps(backtest)

    (
        model_columns,
        block_sizes,
        block_sums,
    ) = aggregate_prediction_date_blocks(model_rps)

    point_estimates = {
        model: float(model_rps[model].mean())
        for model in model_columns
    }

    (
        comparator_indices,
        bootstrap_deltas,
    ) = paired_block_bootstrap(
        model_columns,
        block_sizes,
        block_sums,
    )

    reference_rps = point_estimates[REFERENCE_MODEL]

    rows = []

    for output_index, comparator_index in enumerate(
        comparator_indices
    ):
        comparator = model_columns[comparator_index]
        comparator_rps = point_estimates[comparator]

        delta = reference_rps - comparator_rps

        lower, upper = np.quantile(
            bootstrap_deltas[:, output_index],
            [0.025, 0.975],
        )

        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "comparator_model": comparator,
                "reference_rps": reference_rps,
                "comparator_rps": comparator_rps,
                "delta_rps_reference_minus_comparator": delta,
                "ci_2_5": float(lower),
                "ci_97_5": float(upper),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "prediction_date_blocks": EXPECTED_BLOCKS,
                "held_out_matches": EXPECTED_MATCHES,
                "estimand": "per-match mean RPS",
                "bootstrap_unit": "prediction-date block",
                "seed": RANDOM_SEED,
            }
        )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Held-out matches:", EXPECTED_MATCHES)
    print("Prediction-date blocks:", EXPECTED_BLOCKS)
    print("Bootstrap replicates:", BOOTSTRAP_REPLICATES)
    print("Random seed:", RANDOM_SEED)
    print("Estimand: per-match mean RPS")
    print("Bootstrap unit: prediction-date block")
    print()
    print(
        "Delta convention: "
        "RPS(50/50) - RPS(comparator)."
    )
    print(
        "Negative delta = 50/50 has lower RPS."
    )
    print(
        "Positive delta = comparator has lower RPS."
    )
    print()

    print("Point estimates:")
    for model in model_columns:
        print(
            f"  {model}: "
            f"{point_estimates[model]:.6f}"
        )

    print()
    print("Paired 95% percentile block-bootstrap intervals:")

    for _, row in results.iterrows():
        print(
            "  50/50 vs "
            f"{row['comparator_model']}: "
            f"delta={row['delta_rps_reference_minus_comparator']:.6f}, "
            f"95% CI "
            f"[{row['ci_2_5']:.6f}, "
            f"{row['ci_97_5']:.6f}]"
        )

    print()
    print("Saved:", OUTPUT_PATH)
    print()
    print(
        "Interpretation note: these historical intervals are "
        "conditional on the set of models already compared on the "
        "held-out period. They do not remove the selection optimism "
        "created by choosing Prekick v1 after observing held-out "
        "performance."
    )


if __name__ == "__main__":
    main()
