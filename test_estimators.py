# ============================================================
#  Unit tests for the estimator logic (Steps 1-3).
# ------------------------------------------------------------
#  These tests never touch the ~300 MB dataset. They build tiny synthetic
#  DataFrames whose ITT, LATE, and standard errors can be worked out by hand,
#  then check the code reproduces those numbers. This guards against silent
#  regressions when the analysis scripts are edited, in the spirit of the
#  repo rule "compute every number from the data; verify against a known
#  reference" (see analyze.py's MATCH check).
#
#  Run:  python3 -m unittest test_estimators -v
# ============================================================

import contextlib
import io
import math
import unittest

import pandas as pd

import analyze
import late
import naive

Z95 = late.Z95


@contextlib.contextmanager
def quiet():
    """Silence the functions that print, so test output stays readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class ITTTest(unittest.TestCase):
    """analyze.compute_itt: relative/absolute lift on a hand-built frame."""

    def setUp(self):
        # control: 2/10 convert = 0.20 ; treated: 3/10 convert = 0.30
        treatment = [0] * 10 + [1] * 10
        conversion = ([1, 1] + [0] * 8) + ([1, 1, 1] + [0] * 7)
        self.df = pd.DataFrame(
            {"treatment": treatment, "conversion": conversion})

    def test_rates_and_lift(self):
        r = analyze.compute_itt(self.df, "conversion")
        self.assertAlmostEqual(r["control_rate"], 0.20, places=12)
        self.assertAlmostEqual(r["treated_rate"], 0.30, places=12)
        self.assertAlmostEqual(r["abs_lift"], 0.10, places=12)
        # relative lift = 0.10 / 0.20 = 50%
        self.assertAlmostEqual(r["rel_lift_pct"], 50.0, places=9)


class NaiveTest(unittest.TestCase):
    """naive.conv_rate: conversion rate for a masked subset."""

    def setUp(self):
        self.df = pd.DataFrame({
            "treatment": [0, 0, 1, 1, 1],
            "conversion": [0, 1, 1, 1, 0],
        })

    def test_conv_rate_by_mask(self):
        control = self.df["treatment"] == 0
        treated = self.df["treatment"] == 1
        self.assertAlmostEqual(
            naive.conv_rate(self.df, control), 0.5, places=12)   # 1/2
        self.assertAlmostEqual(
            naive.conv_rate(self.df, treated), 2 / 3, places=12)  # 2/3


class GroupStatsTest(unittest.TestCase):
    """late.group_stats: first/second moments match closed-form values."""

    def test_moments(self):
        # 4 rows: (Y,D) = (1,1),(0,1),(1,0),(0,0)
        sub = pd.DataFrame({"conversion": [1, 0, 1, 0],
                            "exposure":   [1, 1, 0, 0]})
        s = late.group_stats(sub)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["p_y"], 0.5, places=12)
        self.assertAlmostEqual(s["p_d"], 0.5, places=12)
        # P(Y=1 & D=1) = 1/4 = 0.25 ; cov = 0.25 - 0.5*0.5 = 0
        self.assertAlmostEqual(s["cov_bar"], 0.0 / 4, places=12)
        # var of the mean = p(1-p)/n = 0.25/4
        self.assertAlmostEqual(s["var_ybar"], 0.25 / 4, places=12)


class WaldTest(unittest.TestCase):
    """late.wald_confidence_intervals: point estimates and delta-method SEs."""

    def _arms(self, scale=1):
        # Control: exposure all 0, conversion rate 0.10
        n = 1000 * scale
        control = pd.DataFrame({
            "conversion": [1] * (100 * scale) + [0] * (900 * scale),
            "exposure":   [0] * n,
        })
        # Treated: exposure 0.50, conversion 0.20, all converters exposed
        treated = pd.DataFrame({
            "conversion": [1] * (200 * scale) + [0] * (800 * scale),
            "exposure":   [1] * (500 * scale) + [0] * (500 * scale),
        })
        return late.group_stats(treated), late.group_stats(control)

    def test_point_estimates(self):
        t, c = self._arms()
        ci = late.wald_confidence_intervals(t, c)
        # ITT = 0.20 - 0.10 = 0.10 ; first stage = 0.50 ; LATE = 0.10/0.50 = 0.20
        self.assertAlmostEqual(ci["itt"], 0.10, places=12)
        self.assertAlmostEqual(ci["first_stage"], 0.50, places=12)
        self.assertAlmostEqual(ci["late"], 0.20, places=12)

    def test_itt_se_closed_form(self):
        t, c = self._arms()
        ci = late.wald_confidence_intervals(t, c)
        # var = .2*.8/1000 + .1*.9/1000 = (.16+.09)/1000 = .00025
        expected_se = math.sqrt(0.00025)
        se_from_ci = (ci["itt_hi"] - ci["itt"]) / Z95
        self.assertAlmostEqual(se_from_ci, expected_se, places=12)

    def test_ci_is_symmetric(self):
        t, c = self._arms()
        ci = late.wald_confidence_intervals(t, c)
        self.assertAlmostEqual(
            (ci["late_lo"] + ci["late_hi"]) / 2, ci["late"], places=12)
        self.assertLess(ci["late_lo"], ci["late"])
        self.assertGreater(ci["late_hi"], ci["late"])

    def test_se_scales_with_sqrt_n(self):
        # Same rates, 100x rows -> SE should shrink ~10x (1/sqrt(100)).
        t1, c1 = self._arms(scale=1)
        t2, c2 = self._arms(scale=100)
        se1 = (late.wald_confidence_intervals(t1, c1)["late_hi"]
               - late.wald_confidence_intervals(t1, c1)["late"]) / Z95
        se2 = (late.wald_confidence_intervals(t2, c2)["late_hi"]
               - late.wald_confidence_intervals(t2, c2)["late"]) / Z95
        self.assertAlmostEqual(se1 / se2, 10.0, places=1)


class AssumptionTest(unittest.TestCase):
    """late.verify_iv_assumptions: the two data-testable IV checks."""

    def test_one_sided_ok_when_control_unexposed(self):
        with quiet():
            first_stage, one_sided_ok = late.verify_iv_assumptions(0.50, 0.0)
        self.assertAlmostEqual(first_stage, 0.50, places=12)
        self.assertTrue(one_sided_ok)

    def test_flags_always_takers(self):
        # control exposure well above tolerance -> A3 should fail
        with quiet():
            first_stage, one_sided_ok = late.verify_iv_assumptions(0.50, 0.02)
        self.assertAlmostEqual(first_stage, 0.48, places=12)
        self.assertFalse(one_sided_ok)

    def test_relevance_failure_raises(self):
        # no first stage -> cannot form a Wald estimator
        with quiet():
            with self.assertRaises(SystemExit):
                late.verify_iv_assumptions(0.0, 0.0)


class CrossFileConsistencyTest(unittest.TestCase):
    """ITT must be the same quantity whether read via analyze or late."""

    def test_itt_matches_across_files(self):
        control = pd.DataFrame({
            "treatment": 0, "conversion": [1] * 100 + [0] * 900,
            "exposure": 0,
        })
        treated = pd.DataFrame({
            "treatment": 1, "conversion": [1] * 200 + [0] * 800,
            "exposure": [1] * 500 + [0] * 500,
        })
        df = pd.concat([control, treated], ignore_index=True)

        itt_analyze = analyze.compute_itt(df, "conversion")["abs_lift"]
        t = late.group_stats(df[df["treatment"] == 1])
        c = late.group_stats(df[df["treatment"] == 0])
        itt_late = late.wald_confidence_intervals(t, c)["itt"]

        self.assertAlmostEqual(itt_analyze, itt_late, places=12)


if __name__ == "__main__":
    unittest.main()