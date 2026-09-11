# ============================================================
#  Ad Incrementality Analysis - Step 3: LATE (effect on the exposed)
# ------------------------------------------------------------
#  What this file does:
#   Step 1 (ITT) gives the honest effect of being ASSIGNED to ads.
#   Step 2 shows that comparing EXPOSED users directly is badly inflated.
#   This step answers the natural follow-up honestly:
#     "What is the effect for the people who were actually exposed?"
#
#   We use an instrumental-variables (IV) idea. Random assignment
#   (`treatment`) is the INSTRUMENT Z; actual `exposure` is the treatment D
#   whose effect we want; `conversion` is the outcome Y. The Wald estimator
#   is the ITT effect on Y divided by the ITT effect on D (the "first stage"):
#
#       LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])
#            =        ITT (absolute)  /   first-stage exposure gap
#
#   In THIS experiment control users are never shown ads, so E[D|Z=0] = 0 and
#   the denominator collapses to E[D|Z=1] = the compliance (exposed) share of
#   the treated group. We compute the GENERAL denominator from the data and
#   verify that collapse rather than assuming it (see verify_iv_assumptions).
#
#  --- Identifying assumptions (what makes LATE causal) ---
#   (A1) Relevance: the instrument actually moves exposure, i.e. the first
#        stage E[D|Z=1] - E[D|Z=0] > 0. TESTABLE -> checked from the data.
#   (A2) Exclusion restriction: assignment affects conversion ONLY through
#        exposure (being *assigned* but never *shown* an ad does not by itself
#        change buying). NOT testable from data; it rests on the experiment's
#        design -- assignment's only channel to the user is whether an ad is
#        served. Stated here so the estimate is not read as assumption-free.
#   (A3) Monotonicity / one-sided noncompliance: no "defiers", and here the
#        stronger design fact that control users cannot be exposed, i.e. no
#        "always-takers": E[D|Z=0] = 0. TESTABLE -> checked from the data.
#        Under one-sided noncompliance the compliers are exactly the exposed
#        among the treated, so LATE here is the effect ON THE EXPOSED.
#
#  How to run: in the VS Code terminal  ->  python late.py
#  (Run 'python analyze.py' once first so the data is already downloaded.)
# ============================================================

import os

import pandas as pd

DATA_PATH = os.path.join("data", "criteo-uplift-v2.1.csv.gz")

# Largest control-group exposure rate we still treat as "no always-takers".
# Control users are not served ads, so this should be 0; allow a hair of
# tolerance only for data-quirk robustness. Computed value is reported either
# way (MATCH/MISMATCH), never hard-coded into the result.
CONTROL_EXPOSURE_TOL = 1e-4  # 0.01%


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


def verify_iv_assumptions(exposure_treated, exposure_control):
    """Check the two data-testable IV assumptions and return the first-stage
    exposure gap E[D|Z=1] - E[D|Z=0]. Prints a MATCH/MISMATCH line for each,
    in the same spirit as analyze.py's check against published figures."""
    first_stage = exposure_treated - exposure_control

    print("\n  -- IV assumption checks (computed from data) --")

    # (A1) Relevance: instrument must move exposure.
    relevance_ok = first_stage > 0
    print(f"  (A1) Relevance  first stage E[D|Z=1]-E[D|Z=0] = "
          f"{first_stage*100:.2f}pp  "
          f"({'OK' if relevance_ok else 'FAILS: instrument is irrelevant'})")

    # (A3) One-sided noncompliance: control exposure must be ~0 (no always-takers).
    one_sided_ok = exposure_control <= CONTROL_EXPOSURE_TOL
    print(f"  (A3) No always-takers  E[D|Z=0] = {exposure_control*100:.4f}%  "
          f"({'MATCH (~0)' if one_sided_ok else 'MISMATCH (please check)'})")

    # (A2) is a design assumption, not estimable from these columns.
    print("  (A2) Exclusion restriction: design assumption, not testable here.")

    if not relevance_ok:
        raise SystemExit(
            "  Relevance fails: cannot form a Wald estimator (division by <= 0)."
        )
    return first_stage, one_sided_ok


def main():
    print("=" * 56)
    print(" Ad Incrementality Analysis - Step 3: LATE (effect on the exposed)")
    print("=" * 56)
    df = load_data()

    print("\n[2/2] Computing LATE via the Wald estimator...")

    is_control = df["treatment"] == 0
    is_treated = df["treatment"] == 1

    # Outcome rates by assignment (Y)
    p_control = df.loc[is_control, "conversion"].mean()
    p_treated = df.loc[is_treated, "conversion"].mean()

    # ITT in absolute terms (percentage points), same comparison as Step 1
    itt_abs = p_treated - p_control

    # Exposure rates by assignment (D) -- BOTH arms, computed from data
    exposure_treated = df.loc[is_treated, "exposure"].mean()   # compliance
    exposure_control = df.loc[is_control, "exposure"].mean()   # should be ~0

    # Verify the assumptions and get the GENERAL Wald denominator (first stage)
    first_stage, one_sided_ok = verify_iv_assumptions(
        exposure_treated, exposure_control)

    # Wald / LATE using the general denominator (correct whether or not
    # control exposure is exactly 0). When E[D|Z=0]=0 it equals ITT/compliance.
    late_abs = itt_abs / first_stage

    # For context: the naive exposed-vs-control absolute gap (the inflated one)
    p_exposed = df.loc[df["exposure"] == 1, "conversion"].mean()
    naive_abs = p_exposed - p_control

    print(f"\n  Control conversion rate            : {p_control*100:.3f}%")
    print(f"  Treated (assigned) conversion rate : {p_treated*100:.3f}%")
    print(f"  ITT (absolute)                     : {itt_abs*100:.4f} pp")
    print(f"  Compliance E[D|Z=1] (treated exposed): {exposure_treated*100:.2f}%")
    print(f"  Control exposure E[D|Z=0]          : {exposure_control*100:.4f}%")
    print(f"  First-stage denominator            : {first_stage*100:.2f} pp")
    if one_sided_ok:
        print("  (control exposure ~0 -> denominator = compliance, as expected)")
    print("\n  -- Effect for an actually-exposed user (absolute) --")
    print(f"  LATE  (Wald, defensible)           : {late_abs*100:.4f} pp")
    print(f"  Naive (exposed - control, biased)  : {naive_abs*100:.4f} pp")
    print(f"  => Naive overstates the per-exposed effect by "
          f"~{naive_abs/late_abs:.1f}x")

    print("\n" + "=" * 56)
    print(" Interpretation: LATE is the honest answer to 'what did the ad do")
    print(" for those who actually saw it?'. It uses random assignment as an")
    print(" instrument, so it avoids the self-selection that inflates the naive")
    print(" exposed-vs-control comparison. The number is causal only under the")
    print(" assumptions checked above (relevance, exclusion, one-sided")
    print(" noncompliance). ITT (Step 1) remains the effect of assignment;")
    print(" LATE is the effect on compliers - both are honest, the naive is not.")
    print("=" * 56)


if __name__ == "__main__":
    main()