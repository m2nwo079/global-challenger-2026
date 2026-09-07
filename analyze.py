# ============================================================
#  Ad Incrementality Analysis - Step 1: ITT (true effect of ad assignment)
# ------------------------------------------------------------
#  What this file does:
#   (1) Downloads the Criteo incrementality-test dataset from the internet
#   (2) Compares conversion rates between users ASSIGNED to ads vs. control
#   (3) Reports how much the ad actually caused conversions (ITT).
#
#  How to run: in the VS Code terminal  ->  python analyze.py
# ============================================================

import os
import urllib.request

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
# Public data location (currently live: Hugging Face)
DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
# Where to save the downloaded file (a "data" folder inside the project)
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "criteo-uplift-v2.1.csv.gz")

# Published "ground-truth" figures, used to check our own computation
EXPECTED_CONVERSION_LIFT_PCT = 59.45  # relative lift on conversions (%)
EXPECTED_VISIT_LIFT_PCT = 27.07       # relative lift on visits (%)


# ------------------------------------------------------------
# 1. Download the data (only once; skipped if it already exists)
# ------------------------------------------------------------
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH):
        print(f"[1/3] Data already present: {DATA_PATH} (skipping download)")
        return
    print("[1/3] Downloading data... (~300MB, may take a few minutes)")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("      Done!")


# ------------------------------------------------------------
# 2. Load the data
# ------------------------------------------------------------
def load_data():
    print("[2/3] Loading data into memory... (may take 1-2 minutes)")
    # Load only the 4 columns we need (saves memory).
    #  treatment  : assigned to ads? (1 = treated, 0 = control)   <- RANDOMIZED
    #  conversion : did the user convert (purchase)? (1/0)
    #  visit      : did the user visit the site? (1/0)
    #  exposure   : was the user actually exposed to an ad? (1/0)  <- used later
    cols = ["treatment", "conversion", "visit", "exposure"]
    df = pd.read_csv(DATA_PATH, usecols=cols)
    print(f"      Done! {len(df):,} rows total")
    return df


# ------------------------------------------------------------
# 3. Compute ITT: compare the outcome rate of the two groups
# ------------------------------------------------------------
def compute_itt(df, outcome):
    """For a given outcome ('conversion' or 'visit'), compare the treated vs.
    control rate and compute the lift and its statistical significance."""
    # Split by treatment (0/1); get each group's mean (=rate), sum, and count
    g = df.groupby("treatment")[outcome].agg(["mean", "sum", "count"])

    p_ctrl = g.loc[0, "mean"]   # control rate (no ads)
    p_trt = g.loc[1, "mean"]    # treated rate (assigned to ads)

    abs_lift = p_trt - p_ctrl               # absolute difference (pp)
    rel_lift = abs_lift / p_ctrl            # relative lift (percent increase)

    # Test whether the difference in rates is unlikely to be chance (z-test)
    success = [g.loc[1, "sum"], g.loc[0, "sum"]]
    totals = [g.loc[1, "count"], g.loc[0, "count"]]
    _, pvalue = proportions_ztest(success, totals)

    return {
        "outcome": outcome,
        "control_rate": p_ctrl,
        "treated_rate": p_trt,
        "abs_lift": abs_lift,
        "rel_lift_pct": rel_lift * 100,
        "pvalue": pvalue,
    }


def print_result(r, expected_pct):
    print(f"\n  -- Outcome: {r['outcome']} --")
    print(f"  Control rate (no ads)      : {r['control_rate']*100:.3f}%")
    print(f"  Treated rate (ad assigned) : {r['treated_rate']*100:.3f}%")
    print(f"  Absolute lift              : {r['abs_lift']*100:.3f} pp")
    print(f"  Relative lift              : {r['rel_lift_pct']:.2f}%   <- causal effect of ads")
    print(f"  Statistical significance   : p = {r['pvalue']:.2e}"
          f"  ({'significant' if r['pvalue'] < 0.05 else 'not significant'})")
    # Compare against the published figure to verify our computation
    diff = abs(r["rel_lift_pct"] - expected_pct)
    ok = "MATCH" if diff < 1.0 else "MISMATCH (please check)"
    print(f"  vs. published ({expected_pct}%)   : {ok}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=" * 56)
    print(" Ad Incrementality Analysis - Step 1: ITT")
    print("=" * 56)
    download_data()
    df = load_data()

    print("\n[3/3] Computing ITT...")
    conv = compute_itt(df, "conversion")
    visit = compute_itt(df, "visit")
    print_result(conv, EXPECTED_CONVERSION_LIFT_PCT)
    print_result(visit, EXPECTED_VISIT_LIFT_PCT)

    print("\n" + "=" * 56)
    print(" Interpretation: being ASSIGNED to ads alone increased conversions")
    print(" by the relative lift shown above. Because assignment is randomized,")
    print(" this difference is not mere correlation but a CAUSAL effect.")
    print(" This is the honest starting point.")
    print("=" * 56)


if __name__ == "__main__":
    main()