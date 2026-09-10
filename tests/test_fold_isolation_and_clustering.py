"""Regression guards for the corrections made after the independent audit.

Each test here corresponds to a defect that was confirmed and fixed:

  * preprocessing was fitted once on the whole outer-training set and reused
    inside every model-selection and calibration fold;
  * Diabetes-130 subgroup intervals were encounter-level binomial intervals, and
    the gap bootstrap drew an independent multiplicity for a patient appearing
    under more than one demographic label;
  * the reliability diagram plotted nominal bin centres rather than the mean
    prediction of the observations in each bin.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDICAL_FAIRNESS_N_JOBS", "8")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src import figures as figs
from src import uncertainty as unc
from src.pipeline import _model_pipeline
from src.preprocessing import TrainFittedPreprocessor


class ProvenanceFreeTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("MEDICAL_FAIRNESS_PROVENANCE_JSONL", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["MEDICAL_FAIRNESS_PROVENANCE_JSONL"] = self._saved


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num": rng.normal(10.0, 2.0, n),
        "cat": rng.choice(list("abc"), n),
    })
    X.loc[rng.choice(n, n // 6, replace=False), "num"] = np.nan
    y = (rng.random(n) < 0.5).astype(int)
    return X, y


def _pipeline():
    prep = TrainFittedPreprocessor(
        feature_order=["num", "cat"],
        categorical_specs=[{"source": "cat", "prefix": "cat", "drop_first": True}],
    )
    return _model_pipeline(prep, LogisticRegression(max_iter=500))


def _fitted_preprocessing(pipe, data, y, train_idx):
    fitted = clone(pipe).fit(data.iloc[train_idx], y[train_idx])
    step = fitted.named_steps["preprocessing"]
    return dict(step.medians_), {k: tuple(v) for k, v in step.categories_.items()}


class InnerFoldPreprocessingTests(ProvenanceFreeTestCase):
    """1. Validation-fold predictor values must not reach fold preprocessing."""

    def test_validation_rows_cannot_alter_the_fold_training_preprocessor(self):
        X, y = _frame()
        pipe = _pipeline()
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

        for i, (train_idx, val_idx) in enumerate(folds):
            corrupted = X.copy()
            # Corrupt ONLY this fold's validation rows. If any of it reaches the
            # fold's training preprocessor, the median or schema will move.
            corrupted.iloc[val_idx, corrupted.columns.get_loc("num")] = 1e9
            corrupted.iloc[val_idx, corrupted.columns.get_loc("cat")] = "ZZZ"

            before = _fitted_preprocessing(pipe, X, y, train_idx)
            after = _fitted_preprocessing(pipe, corrupted, y, train_idx)
            self.assertEqual(before, after, f"fold {i} preprocessing changed")

    def test_each_fold_learns_its_own_median(self):
        """If one global preprocessor were reused, every fold would match."""
        X, y = _frame()
        pipe = _pipeline()
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
        medians = {
            _fitted_preprocessing(pipe, X, y, tr)[0]["num"] for tr, _ in folds
        }
        self.assertGreater(len(medians), 1)

    def test_unseen_validation_category_creates_no_new_column(self):
        X, y = _frame()
        pipe = _pipeline()
        train_idx = np.arange(0, 300)
        fitted = clone(pipe).fit(X.iloc[train_idx], y[train_idx])
        held = X.iloc[300:].copy()
        held.iloc[0, held.columns.get_loc("cat")] = "UNSEEN"
        transformed = fitted.named_steps["preprocessing"].transform(held)
        self.assertEqual(
            list(transformed.columns),
            list(fitted.named_steps["preprocessing"].feature_names_),
        )


def _clustered(n=1200, n_patients=300, seed=5, span=25):
    """Encounters with repeated patients; `span` patients cross the subgroups."""
    rng = np.random.default_rng(seed)
    patient = np.repeat(np.arange(n_patients), n // n_patients)
    patient = np.concatenate([patient, rng.integers(0, n_patients, n - len(patient))])
    group = np.where(patient % 2 == 0, "A", "B").astype(object)
    crossing = rng.choice(n_patients, span, replace=False)
    for p in crossing:
        rows = np.where(patient == p)[0]
        if len(rows) > 1:
            group[rows[0]] = "A"
            group[rows[1]] = "B"
    prob = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < prob).astype(int)
    pred = (prob >= 0.5).astype(int)
    return y, prob, pred, patient, group


class JointPatientResamplingTests(ProvenanceFreeTestCase):
    """2. One patient, one multiplicity, shared across all their encounters."""

    def test_multiplicity_is_shared_across_all_encounters_of_a_patient(self):
        _, _, _, patient, _ = _clustered()
        positions = np.arange(len(patient))
        keys, pool = unc._cluster_pool(positions, patient)
        rng = np.random.default_rng(0)
        drawn = unc.joint_cluster_positions(rng, keys, pool)

        counts = pd.Series(patient[drawn]).value_counts()
        for cluster, count in counts.items():
            size = len(pool[cluster])
            self.assertEqual(
                count % size, 0,
                f"patient {cluster} appeared {count} times for {size} encounters",
            )

    def test_multiplicity_is_shared_across_subgroups_not_drawn_twice(self):
        """The defect this replaces: independent draws per subgroup."""
        _, _, _, patient, group = _clustered()
        positions = np.arange(len(patient))
        keys, pool = unc._cluster_pool(positions, patient)
        rng = np.random.default_rng(1)
        drawn = unc.joint_cluster_positions(rng, keys, pool)

        spanning = [
            k for k in keys if len({group[p] for p in pool[k]}) > 1
        ]
        self.assertGreater(len(spanning), 0, "fixture has no spanning patients")
        for cluster in spanning:
            rows = pool[cluster]
            taken = int(np.count_nonzero(np.isin(drawn, rows)))
            per_encounter = taken / len(rows)
            self.assertEqual(
                per_encounter, int(per_encounter),
                "a spanning patient's encounters were not drawn together",
            )
            # every encounter of the patient shares one multiplicity
            multiplicities = {
                int(np.count_nonzero(drawn == row)) for row in rows
            }
            self.assertEqual(len(multiplicities), 1, multiplicities)

    def test_spanning_patients_are_counted_and_reported(self):
        y, prob, pred, patient, group = _clustered()
        out = unc.bootstrap_comparison(
            y, prob, pred, group == "A", group == "B",
            cluster_ids=patient, n_boot=50, seed=42,
        )
        self.assertEqual(
            out["design"]["resampling_design"], "joint_patient_cluster_whole_test_set"
        )
        self.assertGreater(out["design"]["n_units_spanning_subgroups"], 0)

    def test_clustered_interval_is_wider_when_encounters_are_correlated(self):
        """Wilson on dependent encounters is anticonservative; that is the point.

        The effect requires genuine within-patient correlation, so the outcome
        here is a property of the PATIENT rather than of the encounter. Without
        such correlation the two intervals are expected to agree closely, which
        is why the fixture builds the correlation explicitly rather than
        assuming clustering always widens an interval.
        """
        from src import metrics as fm

        rng = np.random.default_rng(9)
        n_patients, per_patient = 150, 10
        patient = np.repeat(np.arange(n_patients), per_patient)
        n = len(patient)
        # One latent risk per patient drives every encounter of that patient.
        patient_risk = rng.beta(2.0, 2.0, n_patients)
        prob = np.clip(patient_risk[patient] + rng.normal(0, 0.01, n), 0.01, 0.99)
        y = (rng.random(n) < patient_risk[patient]).astype(int)
        pred = (prob >= 0.5).astype(int)
        group = np.where(patient % 2 == 0, "A", "B")
        mask_a = group == "A"

        out = unc.bootstrap_comparison(
            y, prob, pred, mask_a, group == "B",
            cluster_ids=patient, n_boot=400, seed=42,
        )
        boot = out["subgroup_a"]["sensitivity"]
        wilson = fm.threshold_metric_confidence_intervals(
            fm.threshold_metrics(y[mask_a], pred[mask_a])
        )["sensitivity"]
        self.assertGreater(
            boot["ci_high"] - boot["ci_low"],
            wilson["ci_high"] - wilson["ci_low"],
        )

    def test_row_level_design_is_used_without_clusters(self):
        y, prob, pred, _, group = _clustered()
        out = unc.bootstrap_comparison(
            y, prob, pred, group == "A", group == "B", n_boot=30, seed=42
        )
        self.assertEqual(out["design"]["resampling_unit"], "observation")
        self.assertEqual(out["design"]["n_units_spanning_subgroups"], 0)

    def test_failed_calibration_replicates_are_counted_not_hidden(self):
        y, prob, pred, patient, group = _clustered()
        out = unc.bootstrap_comparison(
            y, prob, pred, group == "A", group == "B",
            cluster_ids=patient, n_boot=40, seed=42,
        )
        slope = out["calibration"]["overall"]["slope"]
        self.assertEqual(
            slope["n_valid_replicates"] + slope["n_failed_replicates"], 40
        )
        self.assertIsInstance(
            out["calibration"]["overall"]["replicate_status_counts"], dict
        )


class ReliabilityDiagramTests(ProvenanceFreeTestCase):
    """8. The x-coordinate must be the observed mean prediction in the bin."""

    def test_x_coordinate_is_the_mean_prediction_not_the_bin_centre(self):
        rng = np.random.default_rng(0)
        n = 4000
        prob = rng.beta(2.0, 8.0, n)          # heavily skewed within bins
        y = (rng.random(n) < prob).astype(int)
        x, obs, counts = figs.calibration_curve_points(y, prob, n_bins=10)

        bins = np.linspace(0.0, 1.0, 11)
        centres = (bins[:-1] + bins[1:]) / 2
        for b in range(10):
            if counts[b] == 0:
                self.assertTrue(np.isnan(x[b]))
                continue
            in_bin = (prob >= bins[b]) & (
                prob < bins[b + 1] if b < 9 else prob <= bins[b + 1]
            )
            self.assertAlmostEqual(x[b], prob[in_bin].mean(), places=10)
            self.assertAlmostEqual(obs[b], y[in_bin].mean(), places=10)

        populated = counts > 0
        self.assertGreater(
            float(np.max(np.abs(x[populated] - centres[populated]))), 1e-3,
            "fixture does not actually distinguish the two conventions",
        )

    def test_empty_bins_are_nan_not_imputed(self):
        prob = np.concatenate([np.full(50, 0.05), np.full(50, 0.95)])
        y = (np.arange(100) % 2).astype(int)
        x, obs, counts = figs.calibration_curve_points(y, prob, n_bins=10)
        self.assertTrue(np.isnan(x[5]))
        self.assertTrue(np.isnan(obs[5]))
        self.assertEqual(counts[5], 0)

    def test_counts_sum_to_the_sample_size(self):
        rng = np.random.default_rng(3)
        prob = rng.uniform(0, 1, 777)
        y = (rng.random(777) < prob).astype(int)
        _, _, counts = figs.calibration_curve_points(y, prob, n_bins=10)
        self.assertEqual(int(counts.sum()), 777)


class GroupedFoldTests(ProvenanceFreeTestCase):
    """4 and 5. No patient may straddle a selection or calibration fold."""

    def test_grouped_folds_never_split_a_patient(self):
        _, _, _, patient, _ = _clustered()
        rng = np.random.default_rng(0)
        y = (rng.random(len(patient)) < 0.4).astype(int)
        X = pd.DataFrame({"num": rng.normal(size=len(patient))})
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        for train_idx, val_idx in splitter.split(X, y, groups=patient):
            overlap = set(patient[train_idx]) & set(patient[val_idx])
            self.assertEqual(overlap, set())


if __name__ == "__main__":
    unittest.main()
