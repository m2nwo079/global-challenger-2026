# ============================================================
#  Ad Incrementality Analysis - Step 3: LATE (effect on the exposed)
# ------------------------------------------------------------
#  What this file does:
#   Step 1 (ITT) gives the honest effect of being ASSIGNED to ads.
#   Step 2 shows that comparing EXPOSED users directly is badly inflated.
#   This step answers the natural follow-up honestly:
#     "What is the effect for the people who were actually exposed?"
#
#   We use an instrumental-variables idea. Random assignment (treatment) is
#   the instrument; actual exposure is the treatment we care about. Only a
#   small subset of assigned users are ever exposed ("compliers"). The Wald
#   estimator recovers the effect for those compliers WITHOUT the self-
#   selection bias of Step 2:
#
#       LATE = ITT (absolute) / compliance rate
#
#   where compliance rate = share of the assigned (treated) group that was
#   actually exposed. Intuitively: the whole assigned group only moved because
#   a few compliers were exposed, so we scale the small assignment effect up
#   by that small exposed share to isolate the per-exposed-person effect.
#
#  How to run: in the VS Code terminal  ->  python late.py
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


def main():
    print("=" * 56)
    print(" Ad Incrementality Analysis - Step 3: LATE (effect on the exposed)")
    print("=" * 56)
    df = load_data()

    print("\n[2/2] Computing LATE via the Wald estimator...")

    is_control = df["treatment"] == 0
    is_treated = df["treatment"] == 1

    p_control = df.loc[is_control, "conversion"].mean()
    p_treated = df.loc[is_treated, "conversion"].mean()

    # ITT in absolute terms (percentage points), same comparison as Step 1
    itt_abs = p_treated - p_control

    # Compliance = share of the treated (assigned) group that was actually exposed
    compliance = df.loc[is_treated, "exposure"].mean()

    # Wald estimator: scale the assignment effect up by the exposed share
    late_abs = itt_abs / compliance

    # For context: the naive exposed-vs-control absolute gap (the inflated one)
    p_exposed = df.loc[df["exposure"] == 1, "conversion"].mean()
    naive_abs = p_exposed - p_control

    print(f"\n  Control conversion rate            : {p_control*100:.3f}%")
    print(f"  Treated (assigned) conversion rate : {p_treated*100:.3f}%")
    print(f"  ITT (absolute)                     : {itt_abs*100:.4f} pp")
    print(f"  Compliance (treated actually exposed): {compliance*100:.2f}%")
    print("\n  -- Effect for an actually-exposed user (absolute) --")
    print(f"  LATE  (Wald, defensible)           : {late_abs*100:.4f} pp")
    print(f"  Naive (exposed - control, biased)  : {naive_abs*100:.4f} pp")
    print(f"  => Naive overstates the per-exposed effect by "
          f"~{naive_abs/late_abs:.1f}x")

    print("\n" + "=" * 56)
    print(" Interpretation: LATE is the honest answer to 'what did the ad do")
    print(" for those who actually saw it?'. It uses random assignment as an")
    print(" instrument, so it avoids the self-selection that inflates the naive")
    print(" exposed-vs-control comparison. ITT (Step 1) remains the effect of")
    print(" assignment; LATE is the effect on compliers - both are honest,")
    print(" the naive number is not.")
    print("=" * 56)


if __name__ == "__main__":
    main()