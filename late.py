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
#  --- Uncertainty (95% confidence intervals) ---
#   ITT is a difference of two proportions; its standard error is the usual
#   sqrt(p_t(1-p_t)/n_t + p_c(1-p_c)/n_c). LATE is the ratio ITT/first-stage,
#   so its standard error comes from the DELTA METHOD, which also accounts for
#   the within-treated correlation between Y (conversion) and D (exposure):
#
#       Var(LATE) ~= Var(N)/D^2 + N^2 Var(D)/D^4 - 2 N Cov(N,D)/D^3
#
#   with N = ITT on Y, D = ITT on D. This is the standard Wald/2SLS SE for a
#   single binary instrument. It is a large-sample normal approximation; it is
#   reliable here (n in the millions) but would be fragile with a weak first
#   stage. All inputs are computed from the data, none are hard-coded.
#
#  How to run: in the VS Code terminal  ->  python late.py
#  (Run 'python analyze.py' once first so the data is already downloaded.)
# ============================================================

import math
import os

import pandas as pd

DATA_PATH = os.path.join("data", "criteo-uplift-v2.1.csv.gz")

# Largest control-group exposure rate we still treat as "no always-takers".
# Control users are not served ads, so this should be 0; allow a hair of
# tolerance only for data-quirk robustness. Computed value is reported either
# way (MATCH/MISMATCH), never hard-coded into the result.
CONTROL_EXPOSURE_TOL = 1e-4  # 0.01%

# Normal quantile for a two-sided 95% interval (a math constant, not data).
Z95 = 1.959964


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


def group_stats(sub):
    """First and second sampling moments for one assignment arm.

    Y = conversion (outcome), D = exposure (treatment we care about).
    Returns each arm's mean rates and the sampling (co)variances of those
    means, so ITT, LATE, and their standard errors are all built from the
    same numbers (each quantity defined once)."""
    n = len(sub)
    p_y = sub["conversion"].mean()
    p_d = sub["exposure"].mean()
    p_yd = (sub["conversion"] * sub["exposure"]).mean()  # P(Y=1 and D=1)
    return {
        "n": n,
        "p_y": p_y,
        "p_d": p_d,
        # sampling variance of each arm's mean, and their covariance
        "var_ybar": p_y * (1 - p_y) / n,
        "var_dbar": p_d * (1 - p_d) / n,
        "cov_bar": (p_yd - p_y * p_d) / n,
    }


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


def wald_confidence_intervals(t, c):
    """Point estimates and 95% CIs for ITT and LATE from the two arms' stats.

    N = ITT on Y = p_y(treated) - p_y(control)
    D = first stage = p_d(treated) - p_d(control)
    LATE = N / D, with a delta-method SE that includes Cov(N, D)."""
    n_hat = t["p_y"] - c["p_y"]                 # ITT (absolute), numerator
    d_hat = t["p_d"] - c["p_d"]                 # first stage, denominator

    var_n = t["var_ybar"] + c["var_ybar"]       # arms independent -> variances add
    var_d = t["var_dbar"] + c["var_dbar"]
    cov_nd = t["cov_bar"] + c["cov_bar"]        # cross-arm cov is 0 by design

    se_itt = math.sqrt(var_n)

    late = n_hat / d_hat
    var_late = (var_n / d_hat**2
                + n_hat**2 * var_d / d_hat**4
                - 2 * n_hat * cov_nd / d_hat**3)
    se_late = math.sqrt(var_late) if var_late > 0 else float("nan")

    return {
        "itt": n_hat, "itt_se": se_itt,
        "itt_lo": n_hat - Z95 * se_itt, "itt_hi": n_hat + Z95 * se_itt,
        "first_stage": d_hat,
        "late": late, "late_se": se_late,
        "late_lo": late - Z95 * se_late, "late_hi": late + Z95 * se_late,
    }


def main():
    print("=" * 56)
    print(" Ad Incrementality Analysis - Step 3: LATE (effect on the exposed)")
    print("=" * 56)
    df = load_data()

    print("\n[2/2] Computing LATE via the Wald estimator...")

    is_control = df["treatment"] == 0
    is_treated = df["treatment"] == 1

    # One pass of moments per arm; everything below is derived from these.
    t = group_stats(df.loc[is_treated, ["conversion", "exposure"]])
    c = group_stats(df.loc[is_control, ["conversion", "exposure"]])

    p_control, p_treated = c["p_y"], t["p_y"]
    exposure_treated, exposure_control = t["p_d"], c["p_d"]

    # Verify the assumptions and get the GENERAL Wald denominator (first stage)
    first_stage, one_sided_ok = verify_iv_assumptions(
        exposure_treated, exposure_control)

    # Point estimates + 95% CIs (delta method for the LATE ratio)
    ci = wald_confidence_intervals(t, c)
    itt_abs, late_abs = ci["itt"], ci["late"]

    # For context: the naive exposed-vs-control absolute gap (the inflated one)
    p_exposed = df.loc[df["exposure"] == 1, "conversion"].mean()
    naive_abs = p_exposed - p_control

    print(f"\n  Control conversion rate            : {p_control*100:.3f}%")
    print(f"  Treated (assigned) conversion rate : {p_treated*100:.3f}%")
    print(f"  ITT (absolute)                     : {itt_abs*100:.4f} pp"
          f"   95% CI [{ci['itt_lo']*100:.4f}, {ci['itt_hi']*100:.4f}]")
    print(f"  Compliance E[D|Z=1] (treated exposed): {exposure_treated*100:.2f}%")
    print(f"  Control exposure E[D|Z=0]          : {exposure_control*100:.4f}%")
    print(f"  First-stage denominator            : {first_stage*100:.2f} pp")
    if one_sided_ok:
        print("  (control exposure ~0 -> denominator = compliance, as expected)")
    print("\n  -- Effect for an actually-exposed user (absolute) --")
    print(f"  LATE  (Wald, defensible)           : {late_abs*100:.4f} pp"
          f"   95% CI [{ci['late_lo']*100:.4f}, {ci['late_hi']*100:.4f}]")
    print(f"  Naive (exposed - control, biased)  : {naive_abs*100:.4f} pp")
    print(f"  => Naive overstates the per-exposed effect by "
          f"~{naive_abs/late_abs:.1f}x")

    print("\n" + "=" * 56)
    print(" Interpretation: LATE is the honest answer to 'what did the ad do")
    print(" for those who actually saw it?'. It uses random assignment as an")
    print(" instrument, so it avoids the self-selection that inflates the naive")
    print(" exposed-vs-control comparison. The number is causal only under the")
    print(" assumptions checked above (relevance, exclusion, one-sided")
    print(" noncompliance); the 95% CIs quantify sampling uncertainty. ITT")
    print(" (Step 1) remains the effect of assignment; LATE is the effect on")
    print(" compliers - both are honest, the naive number is not.")
    print("=" * 56)


if __name__ == "__main__":
    main()