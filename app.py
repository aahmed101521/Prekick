from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

LEDGER_PATH = PROJECT_ROOT / "predictions" / "ledger.csv"
UPCOMING_PATH = PROJECT_ROOT / "data" / "fixtures" / "upcoming_fixtures.csv"


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Prekick",
    page_icon="⚽",
    layout="wide",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Required data file not found: {path}")
        st.stop()

    return pd.read_csv(path)


def most_likely_outcome(row: pd.Series) -> str:
    probabilities = {
        "home": row.get("p_home"),
        "draw": row.get("p_draw"),
        "away": row.get("p_away"),
    }

    if any(pd.isna(value) for value in probabilities.values()):
        return "Not predicted"

    outcome = max(probabilities, key=probabilities.get)

    if outcome == "home":
        return str(row["home_team"])

    if outcome == "away":
        return str(row["away_team"])

    return "Draw"


def format_timestamp(value) -> str:
    if pd.isna(value):
        return "—"

    return value.strftime("%Y-%m-%d %H:%M UTC")


def render_probability(label: str, value) -> None:
    if pd.isna(value):
        st.metric(label, "—")
        st.progress(0)
        return

    st.metric(label, f"{value * 100:.1f}%")
    st.progress(float(value))


def render_fixture_card(row: pd.Series) -> None:
    kickoff = row["kickoff_utc"]

    if pd.isna(kickoff):
        kickoff_text = "Kickoff time unavailable"
    else:
        kickoff_text = kickoff.strftime("%A %d %B · %H:%M UTC")

    fixture = f"{row['home_team']} vs {row['away_team']}"

    with st.container(border=True):
        st.caption(kickoff_text)
        st.subheader(fixture)

        home_col, draw_col, away_col = st.columns(3)

        with home_col:
            render_probability("Home", row["p_home"])

        with draw_col:
            render_probability("Draw", row["p_draw"])

        with away_col:
            render_probability("Away", row["p_away"])

        if all(
            pd.notna(row[column])
            for column in ["p_home", "p_draw", "p_away"]
        ):
            confidence = max(
                row["p_home"],
                row["p_draw"],
                row["p_away"],
            )

            st.caption(
                f"Most likely outcome: **{most_likely_outcome(row)}** "
                f"· Confidence: **{confidence * 100:.1f}%**"
            )
        else:
            st.caption("No prediction stored for this fixture.")


# ---------------------------------------------------------------------
# Load authoritative data
# ---------------------------------------------------------------------

ledger = load_csv(LEDGER_PATH)
upcoming = load_csv(UPCOMING_PATH)


# Parse timestamps

for column in [
    "kickoff_utc",
    "training_cutoff_utc",
    "predicted_at_utc",
]:
    if column in ledger.columns:
        ledger[column] = pd.to_datetime(
            ledger[column],
            utc=True,
            errors="coerce",
        )

if "kickoff_utc" in upcoming.columns:
    upcoming["kickoff_utc"] = pd.to_datetime(
        upcoming["kickoff_utc"],
        utc=True,
        errors="coerce",
    )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("⚽ Prekick")

st.caption(
    "Premier League probabilistic forecasting · "
    "Prekick v1 = 50% Elo + 50% Independent Poisson"
)

st.info(
    "Prekick produces probabilistic forecasts for Premier League matches. "
    "The production model is frozen and predictions recorded in the ledger "
    "are never overwritten."
)


# ---------------------------------------------------------------------
# Overall system status
# ---------------------------------------------------------------------

completed = ledger[ledger["result"].notna()].copy()
pending = ledger[ledger["result"].isna()].copy()

predicted_upcoming = upcoming[
    upcoming["fixture_id"].isin(ledger["fixture_id"])
]

active_matchweek = (
    int(upcoming["matchweek"].min())
    if not upcoming.empty
    else None
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current matchweek",
    active_matchweek if active_matchweek is not None else "—",
)

col2.metric(
    "Upcoming fixtures",
    len(upcoming),
)

col3.metric(
    "Predicted upcoming",
    f"{len(predicted_upcoming)}/{len(upcoming)}",
)

col4.metric(
    "Completed forecasts",
    len(completed),
)


# ---------------------------------------------------------------------
# Current matchweek data
# ---------------------------------------------------------------------

prediction_columns = [
    "fixture_id",
    "model_version",
    "training_cutoff_utc",
    "predicted_at_utc",
    "p_home",
    "p_draw",
    "p_away",
]

current = upcoming.merge(
    ledger[prediction_columns],
    on="fixture_id",
    how="left",
)

current = current.sort_values("kickoff_utc").reset_index(drop=True)

current["Fixture"] = (
    current["home_team"]
    + " vs "
    + current["away_team"]
)

current["Kickoff (UTC)"] = current["kickoff_utc"].dt.strftime(
    "%a %d %b · %H:%M"
)

current["Home"] = current["p_home"] * 100
current["Draw"] = current["p_draw"] * 100
current["Away"] = current["p_away"] * 100

current["Most likely"] = current.apply(
    most_likely_outcome,
    axis=1,
)

current["Confidence"] = (
    current[["p_home", "p_draw", "p_away"]]
    .max(axis=1)
    * 100
)


# ---------------------------------------------------------------------
# Current matchweek presentation
# ---------------------------------------------------------------------

st.divider()

if active_matchweek is not None:
    st.header(f"Matchweek {active_matchweek}")
else:
    st.header("Upcoming Fixtures")

st.caption(
    "Official prospective forecasts stored in the Prekick prediction ledger."
)

if current.empty:
    st.info("No upcoming fixtures are currently available.")
else:
    for start in range(0, len(current), 2):
        left, right = st.columns(2)

        with left:
            render_fixture_card(current.iloc[start])

        if start + 1 < len(current):
            with right:
                render_fixture_card(current.iloc[start + 1])


# ---------------------------------------------------------------------
# Detailed probability table
# ---------------------------------------------------------------------

with st.expander("Detailed probability table"):
    st.dataframe(
        current[
            [
                "Kickoff (UTC)",
                "Fixture",
                "Home",
                "Draw",
                "Away",
                "Most likely",
                "Confidence",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Home": st.column_config.ProgressColumn(
                "Home",
                help="Probability of a home win",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Draw": st.column_config.ProgressColumn(
                "Draw",
                help="Probability of a draw",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Away": st.column_config.ProgressColumn(
                "Away",
                help="Probability of an away win",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence",
                help="Largest of the three outcome probabilities",
                format="%.1f%%",
            ),
        },
    )


# ---------------------------------------------------------------------
# Prediction metadata
# ---------------------------------------------------------------------

with st.expander("Prediction details"):
    metadata = current[
        [
            "Fixture",
            "model_version",
            "predicted_at_utc",
            "training_cutoff_utc",
        ]
    ].copy()

    metadata["predicted_at_utc"] = metadata[
        "predicted_at_utc"
    ].apply(format_timestamp)

    metadata["training_cutoff_utc"] = metadata[
        "training_cutoff_utc"
    ].apply(format_timestamp)

    metadata = metadata.rename(
        columns={
            "model_version": "Model",
            "predicted_at_utc": "Predicted at",
            "training_cutoff_utc": "Training cutoff",
        }
    )

    st.dataframe(
        metadata,
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------
# Live performance
# ---------------------------------------------------------------------

st.divider()
st.header("Live Performance")

st.caption(
    "Performance is calculated only from completed forecasts already stored "
    "in the ledger."
)

model_versions = sorted(
    ledger["model_version"]
    .dropna()
    .unique()
    .tolist()
)

performance_options = model_versions + ["All models"]

if "prekick_v1" in performance_options:
    default_model_index = performance_options.index(
        "prekick_v1"
    )
else:
    default_model_index = 0

selected_model = st.selectbox(
    "Model",
    performance_options,
    index=default_model_index,
)

if selected_model == "All models":
    scored = completed.copy()
else:
    scored = completed[
        completed["model_version"] == selected_model
    ].copy()


if scored.empty:
    st.info(
        f"No completed predictions are available yet for "
        f"{selected_model}."
    )
else:
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)

    score_col1.metric(
        "Scored predictions",
        len(scored),
    )

    score_col2.metric(
        "Mean RPS",
        f"{scored['rps'].mean():.4f}",
    )

    score_col3.metric(
        "Mean Log Loss",
        f"{scored['log_loss'].mean():.4f}",
    )

    score_col4.metric(
        "Mean Brier Score",
        f"{scored['brier'].mean():.4f}",
    )


# ---------------------------------------------------------------------
# Completed predictions
# ---------------------------------------------------------------------

st.divider()
st.header("Completed Predictions")

if completed.empty:
    st.info("No completed predictions are currently stored.")
else:
    completed = completed.sort_values(
        "kickoff_utc",
        ascending=False,
    ).copy()

    completed["Fixture"] = (
        completed["home_team"]
        + " vs "
        + completed["away_team"]
    )

    completed["Score"] = completed.apply(
        lambda row: (
            f"{int(row['home_goals'])}–"
            f"{int(row['away_goals'])}"
        ),
        axis=1,
    )

    completed["Prediction"] = completed.apply(
        most_likely_outcome,
        axis=1,
    )

    completed["Home"] = completed["p_home"] * 100
    completed["Draw"] = completed["p_draw"] * 100
    completed["Away"] = completed["p_away"] * 100

    completed_display = completed[
        [
            "Fixture",
            "model_version",
            "Prediction",
            "Home",
            "Draw",
            "Away",
            "Score",
            "result",
            "rps",
            "log_loss",
            "brier",
        ]
    ].rename(
        columns={
            "model_version": "Model",
            "result": "Result",
            "rps": "RPS",
            "log_loss": "Log Loss",
            "brier": "Brier",
        }
    )

    st.dataframe(
        completed_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Home": st.column_config.NumberColumn(
                "Home",
                format="%.1f%%",
            ),
            "Draw": st.column_config.NumberColumn(
                "Draw",
                format="%.1f%%",
            ),
            "Away": st.column_config.NumberColumn(
                "Away",
                format="%.1f%%",
            ),
            "RPS": st.column_config.NumberColumn(
                "RPS",
                format="%.4f",
            ),
            "Log Loss": st.column_config.NumberColumn(
                "Log Loss",
                format="%.4f",
            ),
            "Brier": st.column_config.NumberColumn(
                "Brier",
                format="%.4f",
            ),
        },
    )


# ---------------------------------------------------------------------
# System details
# ---------------------------------------------------------------------

st.divider()

with st.expander("System status"):
    latest_prediction = ledger[
        "predicted_at_utc"
    ].max()

    latest_training_cutoff = ledger[
        "training_cutoff_utc"
    ].max()

    status = pd.DataFrame(
        {
            "Item": [
                "Ledger rows",
                "Completed predictions",
                "Pending predictions",
                "Upcoming fixtures",
                "Upcoming fixtures with predictions",
                "Latest prediction timestamp",
                "Latest training cutoff",
            ],
            "Value": [
                len(ledger),
                len(completed),
                len(pending),
                len(upcoming),
                len(predicted_upcoming),
                format_timestamp(latest_prediction),
                format_timestamp(latest_training_cutoff),
            ],
        }
    )

    status["Value"] = status["Value"].astype(str)

    st.dataframe(
        status,
        hide_index=True,
        width="stretch",
    )
