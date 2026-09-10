"""Guards for the frozen held-out exports, AUROC intervals, and calibration.

These cover the three capabilities added for the definitive run:

  * calibration intercept/slope must return a documented NA rather than a
    non-converged or separated estimate;
  * AUROC confidence intervals must use the correct resampling unit, be
    reproducible from the seed, and keep whole patients together;
  * the exported predictions must BE the evaluated rows, and the serialized
    objects must reproduce the exported probabilities.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDICAL_FAIRNESS_N_JOBS", "8")

import numpy as np
import pandas as pd

from src import adjustments as adj
from src import frozen_outputs as fo
from src import metrics as fm


class ProvenanceFreeTestCase(unittest.TestCase):
    """Never write into a production run's audit trail from a test."""

    def setUp(self):
        self._saved = os.environ.pop("MEDICAL_FAIRNESS_PROVENANCE_JSONL", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["MEDICAL_FAIRNESS_PROVENANCE_JSONL"] = self._saved


class CalibrationStabilityTests(ProvenanceFreeTestCase):
    def test_well_behaved_sample_reports_ok_status(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.random(4000) < p).astype(int)
        out = fm.calibration_metrics(y, p)
        self.assertEqual(out["calib_status"], "ok")
        self.assertTrue(np.isfinite(out["calib_slope"]))
        self.assertAlmostEqual(out["calib_slope"], 1.0, delta=0.2)

    def test_perfect_separation_returns_na_not_a_huge_slope(self):
        """statsmodels returns a non-converged fit here instead of raising."""
        rng = np.random.default_rng(1)
        p = rng.uniform(0.01, 0.99, 2000)
        y = (p > 0.5).astype(int)          # outcome perfectly ordered by score
        out = fm.calibration_metrics(y, p)
        self.assertIn(out["calib_status"], {"separation", "not_converged"})
        self.assertTrue(np.isnan(out["calib_slope"]))
        self.assertTrue(np.isnan(out["calib_intercept"]))

    def test_single_class_sample_returns_na(self):
        p = np.linspace(0.1, 0.9, 200)
        out = fm.calibration_metrics(np.zeros(200, dtype=int), p)
        self.assertEqual(out["calib_status"], "single_class")
        self.assertTrue(np.isnan(out["calib_slope"]))

    def test_sparse_but_estimable_fit_is_reported_not_suppressed(self):
        """Sparsity alone must not decide reportability.

        An arbitrary event-count cutoff would hide a real estimate. Numerical
        validity decides instead, and sparseness is conveyed by the event count
        and the width of the bootstrap interval.
        """
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 500)
        y = np.zeros(500, dtype=int)
        y[:3] = 1
        out = fm.calibration_metrics(y, p)
        self.assertIn(out["calib_status"],
                      {"ok", "separation", "not_converged"})
        if out["calib_status"] == "ok":
            self.assertTrue(np.isfinite(out["calib_slope"]))

    def test_no_coefficient_magnitude_cutoff_is_applied(self):
        """A large but cleanly converged slope is a real estimate."""
        self.assertFalse(hasattr(fm, "CALIBRATION_MAX_ABS_COEF"))
        self.assertFalse(hasattr(fm, "CALIBRATION_MIN_CLASS_COUNT"))

    def test_constant_probability_is_degenerate_not_fitted(self):
        y = np.array([0, 1] * 100)
        out = fm.calibration_metrics(y, np.full(200, 0.3))
        self.assertEqual(out["calib_status"], "degenerate_predictor")

    def test_clipping_is_counted_and_reported(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.2, 0.8, 400)
        p[:25] = 0.0                       # isotonic can emit exact 0 and 1
        p[25:40] = 1.0
        y = (rng.random(400) < 0.5).astype(int)
        out = fm.calibration_metrics(y, p)
        self.assertEqual(out["calib_n_clipped"], 40)

    def test_brier_is_unchanged_by_the_guards(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 300)
        y = (rng.random(300) < p).astype(int)
        self.assertAlmostEqual(
            fm.calibration_metrics(y, p)["brier"], float(np.mean((p - y) ** 2)), places=12
        )


def _clustered_sample(n=900, n_patients=300, seed=7):
    rng = np.random.default_rng(seed)
    patient = rng.integers(0, n_patients, n)
    # Strong within-patient correlation: the outcome is a patient property.
    patient_effect = rng.normal(size=n_patients)
    score = patient_effect[patient] + rng.normal(scale=0.3, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    prob = 1 / (1 + np.exp(-score))
    group = np.where(patient % 2 == 0, "A", "B")
    return y, prob, patient, group


class AurocIntervalTests(ProvenanceFreeTestCase):
    def test_interval_contains_the_point_estimate(self):
        y, prob, _, _ = _clustered_sample()
        out = adj.bootstrap_auroc_ci(y, prob, n_boot=300, seed=42)
        self.assertLessEqual(out["ci_low"], out["point"])
        self.assertGreaterEqual(out["ci_high"], out["point"])
        self.assertEqual(out["method"], "percentile")
        self.assertEqual(out["resampling_unit"], "observation")

    def test_same_seed_reproduces_the_interval_exactly(self):
        y, prob, _, _ = _clustered_sample()
        a = adj.bootstrap_auroc_ci(y, prob, n_boot=200, seed=42)
        b = adj.bootstrap_auroc_ci(y, prob, n_boot=200, seed=42)
        self.assertEqual(a["ci_low"], b["ci_low"])
        self.assertEqual(a["ci_high"], b["ci_high"])

    def test_point_estimate_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        y, prob, _, _ = _clustered_sample()
        out = adj.bootstrap_auroc_ci(y, prob, n_boot=10, seed=42)
        self.assertAlmostEqual(out["point"], roc_auc_score(y, prob), places=12)

    def test_clustered_paths_are_refused_by_the_superseded_helpers(self):
        """These helpers drew a separate patient pool per subgroup.

        They must not silently reintroduce that design; clustered uncertainty
        goes through uncertainty.bootstrap_comparison.
        """
        y, prob, patient, group = _clustered_sample()
        with self.assertRaises(adj.SupersededResamplingDesign):
            adj.bootstrap_auroc_ci(y, prob, cluster_ids=patient, n_boot=10)
        with self.assertRaises(adj.SupersededResamplingDesign):
            adj.bootstrap_auroc_gap_ci(
                y, prob, group == "A", group == "B",
                cluster_ids=patient, n_boot=10,
            )
        with self.assertRaises(adj.SupersededResamplingDesign):
            adj.bootstrap_gap_ci_clustered(
                y, prob, (prob >= 0.5).astype(int),
                group == "A", group == "B", "sensitivity",
                cluster_ids=patient, n_boot=10,
            )

    def test_the_refusal_names_the_correct_replacement(self):
        with self.assertRaises(adj.SupersededResamplingDesign) as caught:
            adj.bootstrap_gap_ci_clustered()
        self.assertIn("bootstrap_comparison", str(caught.exception))

    def test_row_level_paths_still_work(self):
        y, prob, _, group = _clustered_sample()
        out = adj.bootstrap_auroc_ci(y, prob, n_boot=50, seed=42)
        self.assertEqual(out["resampling_unit"], "observation")
        gap = adj.bootstrap_auroc_gap_ci(
            y, prob, group == "A", group == "B", n_boot=50, seed=42
        )
        self.assertEqual(gap["resampling_unit"], "observation")

    def test_clustered_resample_keeps_a_patient_whole(self):
        _, _, patient, _ = _clustered_sample()
        positions = np.arange(len(patient))
        keys, pool = adj._cluster_index_pool(positions, patient)
        rng = np.random.default_rng(0)
        drawn = adj._resample_positions(rng, positions, keys, pool)
        counts = pd.Series(patient[drawn]).value_counts()
        for cluster, count in counts.items():
            size = len(pool[cluster])
            self.assertEqual(
                count % size, 0,
                f"patient {cluster} appeared {count} times, not a multiple of {size}",
            )

    def test_subgroup_mask_restricts_the_sample(self):
        y, prob, _, group = _clustered_sample()
        out = adj.bootstrap_auroc_ci(y, prob, mask=(group == "A"), n_boot=50, seed=42)
        self.assertEqual(out["n_observations"], int(np.sum(group == "A")))

    def test_gap_point_estimate_is_the_difference_of_subgroup_aurocs(self):
        from sklearn.metrics import roc_auc_score

        y, prob, _, group = _clustered_sample()
        a, b = group == "A", group == "B"
        out = adj.bootstrap_auroc_gap_ci(y, prob, a, b, n_boot=10, seed=42)
        expected = roc_auc_score(y[a], prob[a]) - roc_auc_score(y[b], prob[b])
        self.assertAlmostEqual(out["point"], expected, places=12)

    def test_single_class_replicates_are_counted_not_silently_dropped(self):
        y = np.zeros(60, dtype=int)
        y[:2] = 1                          # resamples will often lose both events
        prob = np.linspace(0, 1, 60)
        out = adj.bootstrap_auroc_ci(y, prob, n_boot=200, seed=42)
        self.assertGreater(out["n_undefined_replicates"], 0)


def _fake_result(n=120, seed=5):
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.01, 0.99, n)
    y = (rng.random(n) < prob).astype(int)
    threshold = 0.4
    return {
        "test_index": np.arange(1000, 1000 + n),
        "g_test": np.where(np.arange(n) % 2 == 0, "Male", "Female"),
        "y_test": y,
        "prob": prob,
        "prob_uncalibrated": np.clip(prob + rng.normal(0, 0.01, n), 0, 1),
        "pred": (prob >= threshold).astype(int),
        "best_model_name": "xgboost",
        "decision_threshold": threshold,
        "n_test": n,
    }


class FrozenPredictionExportTests(ProvenanceFreeTestCase):
    def test_export_is_the_evaluated_rows(self):
        result = _fake_result()
        frame = fo.build_frozen_predictions(
            result, dataset="D", comparison_type="sex", comparison="Male vs Female"
        )
        self.assertEqual(list(frame.columns), list(fo.PREDICTION_COLUMNS))
        np.testing.assert_array_equal(frame["y_true"], result["y_test"])
        np.testing.assert_array_equal(frame["prob_calibrated"], result["prob"])
        np.testing.assert_array_equal(frame["row_id"], result["test_index"])

    def test_row_count_mismatch_is_rejected(self):
        result = _fake_result()
        result["n_test"] = 7
        with self.assertRaises(RuntimeError):
            fo.build_frozen_predictions(
                result, dataset="D", comparison_type="sex", comparison="c"
            )

    def test_predictions_inconsistent_with_the_threshold_are_rejected(self):
        result = _fake_result()
        result["pred"] = 1 - result["pred"]
        with self.assertRaises(RuntimeError):
            fo.build_frozen_predictions(
                result, dataset="D", comparison_type="sex", comparison="c"
            )

    def test_cluster_ids_are_anonymised_but_preserve_grouping_exactly(self):
        result = _fake_result()
        n = len(result["test_index"])
        source = np.repeat(np.arange(2000, 2000 + n // 2), 2)[:n]
        clusters = pd.Series(source, index=pd.Index(result["test_index"]))
        frame = fo.build_frozen_predictions(
            result, dataset="D", comparison_type="sex", comparison="c",
            cluster_ids=clusters,
        )
        exported = frame["cluster_id"].to_numpy()
        # No source identifier survives.
        self.assertFalse(any(str(v) in set(map(str, source)) for v in exported))
        self.assertTrue(all(str(v).startswith("P") for v in exported))
        # Grouping is identical: same partition of rows into clusters.
        def partition(labels):
            groups = {}
            for i, label in enumerate(labels):
                groups.setdefault(label, []).append(i)
            return sorted(tuple(v) for v in groups.values())
        self.assertEqual(partition(exported), partition(source))

    def test_anonymised_labels_are_deterministic(self):
        values = ["b", "a", "b", "c", "a"]
        first, _ = fo.anonymize_clusters(values)
        second, _ = fo.anonymize_clusters(values)
        np.testing.assert_array_equal(first, second)

    def test_missing_cluster_ids_are_rejected(self):
        result = _fake_result()
        clusters = pd.Series([1, 2, 3], index=[1000, 1001, 1002])
        with self.assertRaises(ValueError):
            fo.build_frozen_predictions(
                result, dataset="D", comparison_type="sex", comparison="c",
                cluster_ids=clusters,
            )

    def test_round_trip_through_csv_recovers_the_metrics(self):
        result = _fake_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = fo.write_frozen_predictions(
                result, slug="t", dataset="D", comparison_type="sex",
                comparison="c", results_dir=tmp,
            )
            frame = pd.read_csv(path)
        from sklearn.metrics import roc_auc_score

        self.assertAlmostEqual(
            roc_auc_score(frame["y_true"], frame["prob_calibrated"]),
            roc_auc_score(result["y_test"], result["prob"]),
            places=12,
        )


class SerializationTests(ProvenanceFreeTestCase):
    def test_serialized_objects_reproduce_the_frozen_probabilities(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
        y = (rng.random(200) < 0.5).astype(int)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        prob = model.predict_proba(X)[:, 1]
        result = {
            "best_model_name": "logistic_regression",
            "models": {"logistic_regression": model},
            "calibrated_model": model,
            "preprocessor": None,
            "preprocessing_schema": {},
            "feature_names": list(X.columns),
            "decision_threshold": 0.5,
            "threshold_protocol": "t",
            "calibration_protocol": "c",
            "X_test": X,
            "raw_test": X,
            "prob": prob,
        }
        with tempfile.TemporaryDirectory() as tmp:
            meta = fo.write_fitted_objects(result, slug="t", results_dir=tmp)
        self.assertTrue(meta["serialized"], meta["status"])
        self.assertTrue(meta["verified"], meta["status"])
        self.assertEqual(meta["max_abs_deviation"], 0.0)
        self.assertIsNotNone(meta["sha256"])

    def test_verification_failure_is_reported_not_swallowed(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(50, 2)), columns=list("ab"))
        y = (rng.random(50) < 0.5).astype(int)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        result = {
            "best_model_name": "logistic_regression",
            "models": {"logistic_regression": model},
            "calibrated_model": model,
            "preprocessor": None,
            "preprocessing_schema": {},
            "feature_names": list(X.columns),
            "decision_threshold": 0.5,
            "threshold_protocol": "t",
            "calibration_protocol": "c",
            "X_test": X,
            "raw_test": X,
            "prob": np.zeros(50),          # deliberately wrong
        }
        with tempfile.TemporaryDirectory() as tmp:
            meta = fo.write_fitted_objects(result, slug="t", results_dir=tmp)
        self.assertTrue(meta["serialized"])
        self.assertFalse(meta["verified"])
        self.assertIn("verification_failed", meta["status"])


if __name__ == "__main__":
    unittest.main()
