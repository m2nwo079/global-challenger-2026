# AGENTS.md

Guidance for anyone (human or AI agent) picking up this repository.
**Assume Steps 1-3 are done. Your job is to work on Step 4.**

## What this project is

Answers one question with Criteo's randomized incrementality data:
**"Did the purchase really happen because of the ad?"**

Because `treatment` (ad assignment) is randomized, treated-vs-control
differences are causal, not correlational. Conditioning on `exposure`
(decided after treatment) inflates the effect - the project shows this,
then recovers an honest number.

## Status

| Step | File | What it does | Status |
| ---- | ---- | ------------ | ------ |
| 1 | `analyze.py` | ITT: causal effect of ad assignment | DONE (matches published 59.45% / 27.07%) |
| 2 | `naive.py` | Exposure-conditioning trap (~45x inflation) | DONE |
| 3 | `late.py` | LATE via Wald estimator (effect on the exposed) | DONE |
| 4 | `notebooks/04_heterogeneous_effects.ipynb` | Heterogeneous effects + Qini curves | **IN PROGRESS - your task** |

## Your task (Step 4)

Extend the analysis from "how big is the average effect" to
"**is the effect different across people, and who should we target?**".
Train an uplift model, draw Qini curves, and compare the two targets
`visit` and `conversion`.

The notebook already exists and runs. Improve/finish it - do not rebuild
Steps 1-3.

## How to run

Steps 1-3 (local, fast; use `python3`, not `python`):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 analyze.py    # then naive.py, late.py
```
Step 4 runs on **Google Colab**, not locally (it needs more RAM). Upload
`notebooks/04_heterogeneous_effects.ipynb` and run cells top to bottom.

## Rules (do not break these)

- **Compute every number from the data.** Never hard-code a "measured"
  value into output. Where a published figure exists, compare against it
  and report MATCH/MISMATCH (see `analyze.py`).
- **Define each quantity once.** A metric like "conversion rate" is defined
  in one place and reused - no divergent definitions across files.
- **Never commit the data.** The ~300 MB file is git-ignored; the code
  downloads it at runtime.
- **Keep the flat structure.** Analysis scripts sit at the repo root;
  notebooks go in `notebooks/`. Do not wrap code in a `src/` package.
- **Code and docs in English.** Commit messages in English too.

## Step 4 gotchas

- **No GPU.** Uplift trees use CPU + RAM; a T4 does not help. Do not enable it.
- **RAM is the real limit.** Colab free tier ~12-13 GB. Full 14M rows with a
  heavy model can crash the session. Control load via `SAMPLE_ROWS` and
  `MODEL_SIZE` at the top of the notebook; start small, then scale up.
- **The training cell is the slow one** (it fits 4 models: treated+control x
  visit+conversion). Everything else is quick.
- **`conversion` is rare (~0.3%)**, so its Qini curve is noisier than
  `visit`'s. A weaker conversion curve is expected, not a bug.
- Before committing the notebook, clear its outputs (Edit -> Clear all
  outputs) to keep the repo clean; the Qini figure is saved separately as
  `qini_curves.png`.

## Data

Criteo Uplift Prediction Dataset v2.1 (CC BY-NC-SA 4.0, non-commercial).
Columns: `f0`-`f11` (anonymized features), `treatment` (randomized, 1/0),
`exposure` (post-treatment, 1/0), `visit` (1/0), `conversion` (1/0).
Cite: Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018.
