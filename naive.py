# ============================================================
#  Ad Incrementality Analysis - Step 2: the exposure-conditioning trap
# ------------------------------------------------------------
#  What this file does:
#   Shows how a common WRONG comparison inflates the estimated ad effect.
#   - Correct comparison (ITT):   users ASSIGNED to ads vs. control  (Step 1)
#   - Wrong comparison  (naive):  users actually EXPOSED to ads vs. control
#
#   Exposure is decided AFTER treatment, so exposed users are skewed toward
#   people who were already interested in the brand and visit the site often.
#   Conditioning on exposure therefore mixes "pre-existing propensity" into
#   the "ad effect" and the lift explodes. This is exactly why last-click
#   attribution overstates advertising performance.
#
#  How to run: in the VS Code terminal  ->  python naive.py
#  (Run 'python analyze.py' once first so the data is already downloaded.)
# ============================================================

import os

import pandas as pd

DATA_PATH = os.path.join("data", "criteo-uplift-v2.1.csv.gz")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            "Data file not found. Please run 'python analyze.py' once first."
        )
    print("[1/2] Loading data... (1-2 minutes)")
    cols = ["treatment", "conversion", "exposure"]
    df = pd.read_csv(DATA_PATH, usecols=cols)
    print(f"      Done! {len(df):,} rows total")
    return df


def conv_rate(df, mask):
    """Return the conversion rate for the users selected by `mask`."""
    return df.loc[mask, "conversion"].mean()


def main():
    print("=" * 56)
    print(" Ad Incrementality Analysis - Step 2: exposure-conditioning trap")
    print("=" * 56)
    df = load_data()

    print("\n[2/2] Comparing conversion rates across three groups...")

    # Define the three groups
    is_control = df["treatment"] == 0      # control (no ads)
    is_treated = df["treatment"] == 1      # treated (ASSIGNED to ads)
    is_exposed = df["exposure"] == 1       # actually EXPOSED to an ad

    p_control = conv_rate(df, is_control)
    p_treated = conv_rate(df, is_treated)   # for ITT
    p_exposed = conv_rate(df, is_exposed)   # for the naive (exposure) estimate

    # Relative lift under each approach
    itt_lift = (p_treated / p_control) - 1        # correct value (same as Step 1)
    naive_lift = (p_exposed / p_control) - 1      # inflated value

    # Share of treated users who were actually exposed (shows the skew)
    exposure_rate = df.loc[is_treated, "exposure"].mean()

    # Inflation factor
    inflation = naive_lift / itt_lift

    print(f"\n  Control (no ads)     conversion rate : {p_control*100:.3f}%")
    print(f"  Treated (ad assigned) conversion rate : {p_treated*100:.3f}%")
    print(f"  Exposed (actually shown) conv. rate   : {p_exposed*100:.3f}%")
    print("\n  -- 'Ad effect' under two approaches --")
    print(f"  Correct (ITT, by assignment)   : {itt_lift*100:10.1f}%   <- Step 1 answer")
    print(f"  Wrong  (conditioned on exposure): {naive_lift*100:9.1f}%   <- inflated")
    print(f"\n  Share of treated actually exposed : {exposure_rate*100:.1f}%"
          f"  (this small group is self-selected)")
    print(f"  Inflation factor (wrong / correct): ~{inflation:.0f}x")

    print("\n" + "=" * 56)
    print(" Interpretation: comparing only exposed users inflates the ad effect")
    print(" many times over. Exposure is decided after treatment, so conditioning")
    print(" on it lets 'pre-existing propensity to buy' masquerade as an ad effect.")
    print(" This is why common attribution overstates advertising.")
    print("=" * 56)


if __name__ == "__main__":
    main()