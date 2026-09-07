# global-challenger-2026

Ad incrementality analysis — answering "did the purchase really happen
because of the ad?" using Criteo's randomized incrementality-test data.

## Background

`treatment` (ad assignment) is randomized, so the difference in conversion
rates between the treated and control groups is not mere correlation but the
**causal effect** of advertising. In contrast, actual exposure (`exposure`) is
decided *after* treatment; conditioning on it heavily inflates the estimated
effect. This project surfaces that difference step by step.

## Steps

1. **ITT** — the causal effect of being assigned to ads (`analyze.py`)
2. **Exposure-conditioning trap** — how conditioning on exposure inflates the
   effect (`naive.py`)
3. **LATE** — the effect for users who were actually exposed (planned)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analyze.py     # Step 1
python naive.py       # Step 2
```

The dataset (~300MB) is downloaded automatically on first run. It is large, so
it is not stored in the repository (excluded via `.gitignore`); each user
downloads it via the code.

## License

This repository mixes two things of different nature, so their licenses differ.

- **Code** (the `.py` files, etc.): MIT License — see `LICENSE`.
- **Data** (Criteo Uplift Prediction Dataset v2.1): distributed by Criteo under
  **CC BY-NC-SA 4.0**. This repository does not contain the data; it is
  downloaded from the original source at runtime. Data use is subject to
  Attribution (BY), NonCommercial (NC), and ShareAlike (SA) — i.e. this project
  must be used for **non-commercial purposes only**.

## Data source & citation

Criteo Uplift Prediction Dataset v2.1
(https://huggingface.co/datasets/criteo/criteo-uplift)

> Diemert Eustache, Betlei Artem, Christophe Renaudin, Massih-Reza Amini.
> "A Large Scale Benchmark for Uplift Modeling."
> Proceedings of the AdKDD and TargetAd Workshop, KDD, 2018.