# Prekick Decision Log

This file records important design and methodological decisions made during the development of Prekick. The purpose is to preserve why a decision was made, not just what the current code happens to do.

## 2026-08-19 — Live prediction ledger

**Decision**

Prekick will maintain a public prediction ledger beginning with Matchweek 1 of the 2026/27 Premier League season.

The ledger will be stored at:

`predictions/ledger.csv`

and will be tracked in Git.

Each fixture will have one row containing:

* fixture identifier
* season and matchweek
* kickoff timestamp
* home and away teams
* model version
* training-data cutoff
* prediction timestamp
* probability of a home win
* probability of a draw
* probability of an away win
* result and score fields to be completed after the match
* scoring fields to be completed after the match

The three predicted probabilities must sum to 1.

Published prediction probabilities, prediction timestamps, model versions, and training cutoffs must never be changed after publication. New fixture predictions are added to the ledger. After a fixture is completed, previously empty result and scoring fields may be filled in.

**Reason**

The prospective prediction record is part of the core verification layer of Prekick. Starting from the first match of the season creates an uninterrupted, timestamped record that can later be evaluated independently of model-development choices. Git commit history provides a public record of when predictions were published.
