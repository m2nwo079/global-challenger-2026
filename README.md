# global-challenger-2026

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0.5-150458?style=flat&logo=pandas&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.15.0-4051B5?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![scikit-uplift](https://img.shields.io/badge/scikit--uplift-0.5.1-1D9E75?style=flat)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3DA639?style=flat)

Ad incrementality analysis - answering **"did the purchase really happen
because of the ad?"** using Criteo's randomized incrementality-test data.

## Background

`treatment` (ad assignment) is randomized, so the difference in conversion
rates between the treated and control groups is not correlation but the
**causal effect** of advertising. Actual exposure (`exposure`) is decided
*after* treatment; conditioning on it inflates the estimated effect. This
project surfaces that gap step by step, then recovers an honest number.

## Steps & results

| Step | File | Question | Result |
| ---- | ---- | -------- | ------ |
| 1. ITT | `analyze.py` | Effect of being *assigned* to ads | +59.4% conversions (matches published 59.45%) |
| 2. Naive trap | `naive.py` | What if we compare *exposed* users? | Inflates to +2676% (~45x overstated) |
| 3. LATE | `late.py` | Effect on those *actually exposed* | Recovered without the inflation (Wald estimator) |
| 4. Heterogeneity | `notebooks/04_heterogeneous_effects.ipynb` | *Who* should we target? | Qini curves; targeting the top-ranked users captures most of the gain |

**Takeaway:** ads do cause conversions, but the common "exposed-vs-control"
comparison overstates the effect by up to ~45x. Random assignment, used as
an instrument, recovers the honest effect - and an uplift model shows the
effect is concentrated in a targetable minority of users.

## How to run

Steps 1-3 run locally (fast). Use `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 analyze.py    # Step 1: ITT
python3 naive.py      # Step 2: exposure-conditioning trap
python3 late.py       # Step 3: LATE
```

The dataset (~300 MB) downloads automatically on first run. It is not stored
in the repository (excluded via `.gitignore`).

Step 4 runs on **Google Colab** (it needs more RAM). Upload
`notebooks/04_heterogeneous_effects.ipynb` and run the cells top to bottom.
No GPU required. The notebook installs `scikit-learn==1.3.2` and
`scikit-uplift==0.5.1` itself.

## License

- **Code**: MIT License - see `LICENSE`.
- **Data** (Criteo Uplift Prediction Dataset v2.1): distributed by Criteo
  under **CC BY-NC-SA 4.0**. Not included in this repository; downloaded from
  the original source at runtime. Non-commercial use only.

## Data source & citation

Criteo Uplift Prediction Dataset v2.1
(https://huggingface.co/datasets/criteo/criteo-uplift)

> Diemert Eustache, Betlei Artem, Christophe Renaudin, Massih-Reza Amini.
> "A Large Scale Benchmark for Uplift Modeling."
> Proceedings of the AdKDD and TargetAd Workshop, KDD, 2018.
