# Prekick

**Prekick** is a probabilistic forecasting project for English Premier League football matches.

Rather than predicting a single winner, Prekick estimates three probabilities for every fixture:

* **P(Home win)**
* **P(Draw)**
* **P(Away win)**

The project is designed around transparent statistical modelling, leakage-safe evaluation, and reproducible live forecasting.

The current production model is:

> **Prekick v1 = 50% Elo + 50% Independent Poisson**

The betting market is retained as an external benchmark and is not used as an input to Prekick v1.

---

## Project goals

Prekick aims to build a football forecasting system that is:

* probabilistic rather than deterministic;
* evaluated on genuinely unseen matches;
* explicit about model-selection decisions;
* protected against look-ahead bias;
* simple enough to inspect and understand;
* capable of producing and recording live predictions before matches are played.

The project currently focuses exclusively on the **English Premier League**.

---

## Data

The historical modelling dataset covers five EPL seasons:

```text
2021/22
2022/23
2023/24
2024/25
2025/26
```

This gives:

```text
1900 historical matches
2021-08-13 to 2026-05-24
```

Historical match-result distribution:

| Result   | Matches | Proportion |
| -------- | ------: | ---------: |
| Home win |     839 |     0.4416 |
| Draw     |     454 |     0.2389 |
| Away win |     607 |     0.3195 |

The frozen historical modelling data are stored separately from live 2026/27 results so that the original backtest dataset is not modified during live forecasting.

Current-season completed matches are stored in:

```text
data/fixtures/completed_matches_2026_27.csv
```

Upcoming fixtures are stored in:

```text
data/fixtures/upcoming_fixtures.csv
```

---

## Evaluation philosophy

A central requirement of Prekick is that models must be evaluated using information that would actually have been available before each fixture.

For each prediction date:

```python
train_data = data[data["Date"] < prediction_date]
predict_data = data[data["Date"] == prediction_date]
```

All matches played on the same date are predicted together.

This prevents results from earlier matches on a given date from leaking into predictions for later matches on the same date.

### Protected validation period

Model-selection decisions were made using:

```text
2022-03-01 to 2022-05-22
```

This contains:

```text
124 matches
42 prediction-date blocks
```

### Held-out backtest period

Final model evaluation uses:

```text
2022-08-05 to 2026-05-24
```

This contains:

```text
1520 matches
460 prediction-date blocks
```

The held-out period is not used for tuning model hyperparameters or ensemble weights.

---

## Evaluation metrics

Forecasts are evaluated using three probabilistic scoring rules.

### Ranked Probability Score

RPS accounts for the ordered nature of football outcomes and is the primary model-comparison metric used in this project.

Lower values are better.

### Log Loss

Log Loss strongly penalizes confident predictions assigned to outcomes that do not occur.

Lower values are better.

### Brier Score

The multiclass Brier Score measures squared error between forecast probabilities and the observed result.

Lower values are better.

---

## Models

Several forecasting approaches have been implemented and evaluated.

### Base rate

A simple historical result-frequency benchmark.

It predicts approximately:

```text
Home: 0.4416
Draw: 0.2389
Away: 0.3195
```

This establishes a minimum benchmark that more sophisticated models should beat.

---

### Elo

The Elo model tracks changing team strength over time.

Current core settings:

```python
INITIAL_RATING = 1500
HOME_ADVANTAGE = 100
K_FACTOR = 20
```

A separate draw parameter is fitted to convert the underlying Elo strength comparison into Home/Draw/Away probabilities.

Historical Elo states are stored in:

```text
data/processed/elo_history.csv
```

---

### Independent Poisson

The Poisson model estimates team-specific attacking and defensive strength.

Expected home and away goals are calculated from:

* home-team attack;
* home-team defence;
* away-team attack;
* away-team defence;
* home advantage.

Independent Poisson scoreline probabilities are then aggregated into:

```text
P(Home)
P(Draw)
P(Away)
```

---

### Dixon-Coles

A Dixon-Coles adjustment was also evaluated to account for dependence among low-scoring football results.

In the current data, the fitted dependence parameter was close to zero and produced almost no improvement over the standard independent Poisson model.

---

### Multinomial model

A regularized multinomial logistic model was developed using lagged team-performance features.

The model uses 28 predictors based on recent:

* goals;
* goals conceded;
* shots;
* shots conceded;
* shots on target;
* shots on target conceded;
* available match-history indicators.

Continuous features are standardized using training-period statistics only.

Missing lag values are imputed using training-period means before prediction.

The regularization parameter was selected using the protected validation period and then frozen before held-out evaluation.

The final penalty strength is:

```python
penalty_strength = 10.0
```

The multinomial model improves substantially over the base-rate benchmark but does not outperform Elo or Poisson.

---

## Model comparison

Held-out results:

| Model                                     |          RPS |     Log Loss |        Brier |
| ----------------------------------------- | -----------: | -----------: | -----------: |
| Market benchmark                          | **0.194797** | **0.960498** | **0.570333** |
| **Elo + Poisson 50/50**                   | **0.202809** | **0.985390** | **0.587317** |
| Validation-selected Elo 25% / Poisson 75% |     0.203784 |     0.987844 |     0.589174 |
| Elo                                       |     0.204406 |     0.990971 |     0.591136 |
| Dixon-Coles                               |     0.205927 |     0.994733 |     0.593505 |
| Independent Poisson                       |     0.205940 |     0.994614 |     0.593542 |
| Multinomial                               |     0.220859 |     1.043945 |     0.626485 |
| Base rate                                 |     0.231623 |     1.068443 |     0.646272 |

The strongest non-market model in the held-out backtest is the fixed 50/50 Elo-Poisson ensemble.

This therefore becomes:

> **Prekick v1**

The ensemble weight will not be retuned using held-out or live results.

---

## Why the 50/50 ensemble?

Elo and Poisson capture different aspects of team strength.

Elo represents changing overall competitive strength through match results.

Poisson models scoring rates through attacking and defensive parameters.

Their errors are therefore not identical.

A simple equal-weight ensemble:

```text
P(Prekick) = 0.50 × P(Elo) + 0.50 × P(Poisson)
```

performed better on the held-out period than either component model individually.

Although a 25% Elo / 75% Poisson ensemble achieved the best RPS during the earlier validation period, the fixed 50/50 model subsequently produced the strongest non-market held-out performance.

No weight was retuned after observing held-out results.

---

## Betting-market benchmark

Closing bookmaker probabilities are used as an external benchmark.

Preferred odds:

```text
PSCH
PSCD
PSCA
```

If Pinnacle closing odds are unavailable, average closing odds are used:

```text
AvgCH
AvgCD
AvgCA
```

Bookmaker margins are removed by:

1. converting odds to inverse implied probabilities;
2. normalizing the three values so they sum to one.

These probabilities are used **only for comparison**.

They are not features or inputs to Prekick v1.

The market currently remains stronger than the statistical models in the held-out evaluation.

---

## Live forecasting

The project has now moved from historical backtesting into prospective forecasting for the **2026/27 Premier League season**.

Live model training combines:

```text
1900 frozen historical matches
+
completed 2026/27 matches available before prediction
```

For the current Matchweek 3 prediction checkpoint:

```text
Historical matches:        1900
2026/27 completed matches:   20
Live training matches:      1920
Training cutoff:      2026-08-31
```

Both Elo and Poisson are refitted/reconstructed using only matches completed before the upcoming fixtures.

---

## Prediction ledger

Live predictions are stored in:

```text
predictions/ledger.csv
```

Each row records:

```text
fixture_id
season
matchweek
kickoff_utc
home_team
away_team
model_version
training_cutoff_utc
predicted_at_utc
p_home
p_draw
p_away
home_goals
away_goals
result
rps
log_loss
brier
```

The ledger separates prediction-time information from information added after a match is completed.

Prediction fields are treated as immutable.

Once a prediction has been recorded, the forecasting script will refuse to insert another row with the same fixture ID.

Results and scoring metrics are added later without replacing the original probabilities.

---

## Current live predictions

The first official `prekick_v1` predictions have been recorded for Matchweek 3 of the 2026/27 season.

| Fixture                    |  Home |  Draw |  Away |
| -------------------------- | ----: | ----: | ----: |
| Ipswich vs Liverpool       | 0.164 | 0.191 | 0.646 |
| Newcastle vs Bournemouth   | 0.500 | 0.237 | 0.263 |
| Brentford vs Sunderland    | 0.480 | 0.258 | 0.261 |
| Brighton vs Leeds          | 0.578 | 0.221 | 0.201 |
| Fulham vs Crystal Palace   | 0.466 | 0.256 | 0.278 |
| Man City vs Coventry City  | 0.786 | 0.147 | 0.067 |
| Nott'm Forest vs Tottenham | 0.455 | 0.238 | 0.307 |
| Hull City vs Aston Villa   | 0.461 | 0.269 | 0.270 |
| Everton vs Man United      | 0.355 | 0.265 | 0.379 |
| Arsenal vs Chelsea         | 0.621 | 0.214 | 0.164 |

These probabilities were generated using data available through:

```text
2026-08-31
```

and were written to the live ledger before the fixtures were played.

---

## Repository structure

Important project locations currently include:

```text
data/
    processed/
        model_data.csv
        elo_history.csv
    fixtures/
        completed_matches_2026_27.csv
        upcoming_fixtures.csv

predictions/
    ledger.csv

scripts/
    build_elo_history.py
    generate_prekick_v1_predictions.py
    score_live_ledger.py
    walk_forward_poisson_backtest.py
    walk_forward_multinomial_backtest.py

src/
    prekick/
        elo.py
        goal_model.py
        poisson.py
        multinomial.py
        preprocessing.py
        scoring.py

tests/
```

Additional research and backtesting scripts are retained in the repository as the project develops.

---

## Testing

The project uses `pytest`.

Run the complete test suite with:

```powershell
pytest
```

Current status:

```text
91 passed
```

The tests cover the core statistical and production components including:

* Elo;
* Poisson probabilities;
* goal-model fitting;
* Dixon-Coles adjustments;
* ensembles;
* multinomial modelling;
* preprocessing;
* probabilistic scoring;
* live training-count validation;
* fixture and prediction-count validation;
* live-ledger duplicate protection.

---

## Live prediction script

Current Prekick v1 predictions are generated with:

```powershell
python scripts\generate_prekick_v1_predictions.py
```

The script:

1. loads the frozen historical data;
2. adds all completed current-season results available before the prediction batch;
3. reconstructs the current Elo state;
4. verifies the historical Elo reconstruction against the stored Elo history;
5. fits the current Independent Poisson model;
6. loads and validates the upcoming fixtures;
7. identifies any previously unseen teams;
8. generates Elo probabilities;
9. generates Poisson probabilities;
10. combines them using the fixed 50/50 Prekick v1 ensemble;
11. validates the prediction count and probability outputs;
12. prepares immutable prediction-ledger rows;
13. checks whether any fixture IDs already exist in the ledger;
14. appends only valid new prediction rows.

Match and fixture counts are dynamic. The script therefore does not assume a fixed number of completed matches or upcoming fixtures for a matchweek.

A repeated attempt to generate predictions for fixture IDs that are already present in the ledger is rejected. This prevents an official prediction from being silently overwritten or regenerated after it has already been recorded.

---

## Weekly production workflow

The live workflow is designed to keep prospective predictions separate from later match outcomes.

For each new Premier League prediction batch:

### 1. Update completed matches

Add newly completed 2026/27 Premier League matches to:

```text
data/fixtures/completed_matches_2026_27.csv
```

Only matches that were completed before the upcoming prediction batch should be included.

The completed-match data become part of the training history used to reconstruct the current Elo state and refit the Poisson model.

### 2. Update upcoming fixtures

Replace the contents of:

```text
data/fixtures/upcoming_fixtures.csv
```

with the fixtures that should receive the next official Prekick v1 predictions.

Each fixture must have a unique `fixture_id`.

### 3. Generate official predictions

Run:

```powershell
python scripts\generate_prekick_v1_predictions.py
```

The script fits the live Elo and Poisson states using all permitted completed matches, generates the fixed Prekick v1 probabilities, validates them, and records them in:

```text
predictions/ledger.csv
```

Once a fixture has been recorded in the ledger, rerunning the generator with the same fixture ID is rejected.

### 4. Preserve the prediction record

The following ledger fields represent the original prospective forecast and should remain unchanged after prediction time:

```text
fixture_id
season
matchweek
kickoff_utc
home_team
away_team
model_version
training_cutoff_utc
predicted_at_utc
p_home
p_draw
p_away
```

Match outcomes and evaluation metrics are added only after the relevant matches have been completed.

### 5. Add completed outcomes to the ledger

After the fixtures are completed, populate the corresponding result fields in the ledger:

```text
home_goals
away_goals
result
```

where `result` is:

```text
H
D
A
```

for home win, draw, or away win.

The original probability and prediction metadata must not be changed.

### 6. Score completed predictions

Run:

```powershell
python scripts\score_live_ledger.py
```

The scoring script calculates:

```text
RPS
Log Loss
Brier Score
```

for completed predictions while verifying that the immutable prediction fields have not been altered.

### 7. Run the test suite

Before committing a weekly update, run:

```powershell
pytest
```

All tests should pass before the update is committed.

### 8. Commit the weekly checkpoint

The updated data, predictions, results, and scores can then be committed as a new reproducible project checkpoint.

---

## Scoring live predictions

Completed live predictions can be scored with:

```powershell
python scripts\score_live_ledger.py
```

The scoring process calculates:

```text
RPS
Log Loss
Brier Score
```

while checking that the original prediction fields have not been modified.

This makes the live ledger a prospective evaluation record rather than a retrospectively reconstructed prediction table.

---

## Current project status

Prekick v1 is complete as a first production forecasting system.

Its main statistical decisions are frozen:

```text
Model:          50% Elo + 50% Independent Poisson
League:         English Premier League
Primary metric: Ranked Probability Score
Market data:    External benchmark only
```

The historical model-selection and held-out evaluation stages are complete. The live prediction generator uses dynamic match and fixture counts, prediction-ledger duplicate protection is implemented, and the live workflow has dedicated automated tests.

The remaining activity is prospective operation rather than model development:

* add newly completed Premier League results;
* generate each new fixture batch before kickoff;
* preserve the original prediction probabilities;
* score predictions after results become available;
* monitor live performance over a growing prospective sample.

Prekick v1 should not be retuned in response to individual live results. Any future model changes should be treated as a separately versioned model rather than retroactively modifying Prekick v1.

---

## Reproducibility and modelling principles

Prekick follows several rules intended to keep its evaluation credible.

### No future information

Predictions use only information available before the relevant fixture.

### No held-out retuning

Model choices are not changed because of performance observed on the held-out backtest.

### No live-result retuning

Prekick v1 will not be adjusted simply because a small number of live predictions perform unusually well or poorly.

### Market separation

Bookmaker probabilities remain an external reference rather than a model feature.

### Frozen historical backtest data

New 2026/27 results are not inserted into the historical dataset used for the original backtests.

### Immutable live predictions

Once recorded, live prediction probabilities are preserved so that subsequent evaluation reflects what was genuinely predicted before kickoff.

---

## Development

Prekick is being developed incrementally with small, tested checkpoints.

The repository uses Git for version control, with modelling, evaluation, and live-prediction changes committed separately where practical.

Repository:

```text
https://github.com/aahmed101521/Prekick
```

---

## Disclaimer

Prekick is a statistical modelling and research project.

Its forecasts are estimates of uncertainty, not guarantees of match outcomes.

The project is not intended to provide financial or betting advice.
