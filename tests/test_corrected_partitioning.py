"""Phase 7 validation for the corrected analysis path.

These tests are synthetic: they never read or write production results. They
assert the two corrections actually hold, plus the leakage properties that
were already correct and must not regress.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src import datasets as ds
from src.pipeline import (
    _assert_no_cluster_leakage_in_folds,
    run_fairness_analysis,
)
from src.preprocessing import TrainFittedPreprocessor


class ProvenanceFreeTestCase(unittest.TestCase):
    """Base case that keeps synthetic test analyses out of run provenance.

    ``run_fairness_analysis`` appends an analytical provenance record
    whenever MEDICAL_FAIRNESS_PROVENANCE_JSONL is set. Production runs set
    it for every command, including the pre/postflight unit tests, so
    without this guard synthetic fixtures would be written into the run's
    audit trail as if they were real analyses.
    """

    def setUp(self):
        environment = {
            k: v for k, v in os.environ.items()
            if k != "MEDICAL_FAIRNESS_PROVENANCE_JSONL"
        }
        self._env_patch = patch.dict(os.environ, environment, clear=True)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        super().setUp()


def _synthetic_clustered_frame(n_patients=240, seed=0):
    """Encounters nested in patients, with a stable per-patient sex."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        sex = "Male" if patient % 2 == 0 else "Female"
        for _ in range(rng.integers(1, 4)):
            rows.append({
                "patient_nbr": patient,
                "gender": sex,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "race": rng.choice(["A", "B", "C"]),
                "y": int(rng.random() < 0.3),
            })
    frame = pd.DataFrame(rows)
    # Guarantee both classes are present for stratified machinery.
    frame.loc[frame.index[:20], "y"] = 1
    frame.loc[frame.index[-20:], "y"] = 0
    return frame


class TestPreprocessingIsTrainingOnly(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.frame = pd.DataFrame({
            "num": np.concatenate([np.full(50, 1.0), np.full(50, 100.0)]),
            "cat": ["x"] * 50 + ["y"] * 50,
        })
        self.frame.loc[[0, 51], "num"] = np.nan
        self.train_idx = self.frame.index[:50]
        self.test_idx = self.frame.index[50:]
        self.rng = rng

    def test_imputation_statistic_ignores_test_rows(self):
        spec = [{"source": "cat", "prefix": "cat", "drop_first": True}]
        pre = TrainFittedPreprocessor(["num", "cat"], spec)
        pre.fit(self.frame.loc[self.train_idx])
        learned = pre.medians_["num"]

        # Mutating test rows arbitrarily must not move the learned median.
        mutated = self.frame.copy()
        mutated.loc[self.test_idx, "num"] = 999999.0
        pre2 = TrainFittedPreprocessor(["num", "cat"], spec)
        pre2.fit(mutated.loc[self.train_idx])
        self.assertEqual(learned, pre2.medians_["num"])
        # And the training median must equal the training-only median.
        self.assertEqual(
            learned, float(self.frame.loc[self.train_idx, "num"].median())
        )

    def test_test_only_category_creates_no_new_feature(self):
        frame = self.frame.copy()
        frame.loc[self.test_idx, "cat"] = "TEST_ONLY_LEVEL"
        spec = [{"source": "cat", "prefix": "cat", "drop_first": False}]
        pre = TrainFittedPreprocessor(["num", "cat"], spec)
        pre.fit(frame.loc[self.train_idx])
        train_mat = pre.transform(frame.loc[self.train_idx], partition="train")
        test_mat = pre.transform(frame.loc[self.test_idx], partition="test")

        self.assertNotIn("cat_TEST_ONLY_LEVEL", test_mat.columns)
        self.assertEqual(list(train_mat.columns), list(test_mat.columns))
        # Unseen rows fall back to all-zero indicators and are recorded.
        self.assertEqual(
            test_mat[[c for c in test_mat.columns if c.startswith("cat_")]]
            .to_numpy().sum(),
            0.0,
        )
        self.assertIn("test", pre.unseen_category_counts_)

    def test_matrices_share_ordered_columns_and_have_no_nans(self):
        spec = [{"source": "cat", "prefix": "cat", "drop_first": True}]
        pre = TrainFittedPreprocessor(["num", "cat"], spec)
        pre.fit(self.frame.loc[self.train_idx])
        train_mat = pre.transform(self.frame.loc[self.train_idx])
        test_mat = pre.transform(self.frame.loc[self.test_idx])
        self.assertEqual(list(train_mat.columns), list(test_mat.columns))
        self.assertEqual(list(train_mat.columns), pre.feature_names_)
        self.assertFalse(train_mat.isnull().to_numpy().any())
        self.assertFalse(test_mat.isnull().to_numpy().any())

    def test_categorical_source_position_is_preserved(self):
        spec = [{"source": "cat", "prefix": "cat", "drop_first": True}]
        pre = TrainFittedPreprocessor(["cat", "num"], spec).fit(
            self.frame.loc[self.train_idx]
        )
        self.assertEqual(pre.feature_names_[-1], "num")


class TestLoadersDeferDataDerivedPreprocessing(unittest.TestCase):
    """The loaders must no longer impute or dummy before the split."""

    def test_no_loader_imputes_or_dummies(self):
        import inspect
        source = inspect.getsource(ds)
        self.assertNotIn("pd.get_dummies", source)
        self.assertNotIn("fillna(df[col].median())", source)

    def test_diabetes130_keeps_patient_nbr_out_of_predictors(self):
        import inspect
        source = inspect.getsource(ds.load_diabetes130)
        self.assertIn('cluster_col = "patient_nbr"', source)
        # patient_nbr must not be added to the feature list.
        self.assertNotIn('"patient_nbr"] + ', source)


class TestGroupAwarePartitioning(ProvenanceFreeTestCase):
    def setUp(self):
        super().setUp()
        self.frame = _synthetic_clustered_frame()

    def test_group_shuffle_split_has_zero_patient_overlap(self):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_pos, test_pos = next(
            splitter.split(self.frame, self.frame["y"],
                           groups=self.frame["patient_nbr"])
        )
        train_patients = set(self.frame.iloc[train_pos]["patient_nbr"])
        test_patients = set(self.frame.iloc[test_pos]["patient_nbr"])
        self.assertEqual(len(train_patients & test_patients), 0)

    def test_selection_and_calibration_folds_have_zero_patient_overlap(self):
        folds = list(
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            .split(self.frame, self.frame["y"], groups=self.frame["patient_nbr"])
        )
        clusters = self.frame["patient_nbr"].to_numpy()
        self.assertTrue(
            _assert_no_cluster_leakage_in_folds(folds, clusters, "test")
        )

    def test_fold_leakage_assertion_actually_fires(self):
        positions = np.arange(len(self.frame))
        bad = [(positions, positions)]  # same rows both halves
        with self.assertRaises(RuntimeError):
            _assert_no_cluster_leakage_in_folds(
                bad, self.frame["patient_nbr"].to_numpy(), "test"
            )

    def test_pipeline_reports_zero_overlap_and_excludes_cluster_id(self):
        result = run_fairness_analysis(
            df=self.frame,
            feature_cols=["x1", "x2", "race"],
            target_col="y",
            group_col="gender",
            group_a_value="Male",
            group_b_value="Female",
            threshold="prevalence",
            n_boot=20,
            cluster_ids=self.frame["patient_nbr"],
            preprocess_spec={
                "categorical": [
                    {"source": "race", "prefix": "race", "drop_first": True}
                ]
            },
            run_calibration_adjustment=False,
            run_case_mix=False,
            verbose=False,
        )
        diagnostics = result["split_diagnostics"]
        self.assertEqual(diagnostics["cluster_overlap_train_test"], 0)
        self.assertTrue(diagnostics["grouped_by_cluster"])
        self.assertNotIn("patient_nbr", result["feature_names"])
        self.assertEqual(
            result["bootstrap_protocol"],
            "patient_cluster_within_subgroup_resampling",
        )
        self.assertEqual(
            result["calibration_protocol"],
            "five_fold_isotonic_fold_ensemble_patient_grouped",
        )
        # Independent re-derivation of the overlap claim.
        train_patients = set(self.frame.loc[result["train_index"], "patient_nbr"])
        test_patients = set(self.frame.loc[result["test_index"], "patient_nbr"])
        self.assertEqual(len(train_patients & test_patients), 0)


class TestNoTestOutcomeInfluenceBeforeEvaluation(ProvenanceFreeTestCase):
    """Properties that were already correct and must not regress."""

    def setUp(self):
        super().setUp()
        self.frame = _synthetic_clustered_frame(seed=5)

    def _run(self, frame):
        return run_fairness_analysis(
            df=frame,
            feature_cols=["x1", "x2", "race"],
            target_col="y",
            group_col="gender",
            group_a_value="Male",
            group_b_value="Female",
            threshold="prevalence",
            n_boot=0,
            cluster_ids=frame["patient_nbr"],
            preprocess_spec={
                "categorical": [
                    {"source": "race", "prefix": "race", "drop_first": True}
                ]
            },
            run_calibration_adjustment=False,
            run_case_mix=False,
            verbose=False,
        )

    def test_model_selection_calibration_and_threshold_ignore_test_outcomes(self):
        baseline = self._run(self.frame)

        # Flip every TEST outcome. Model selection, the fitted/calibrated
        # model, and the primary threshold must all be unchanged, because
        # none of them may consult test outcomes.
        perturbed = self.frame.copy()
        perturbed.loc[baseline["test_index"], "y"] = (
            1 - perturbed.loc[baseline["test_index"], "y"]
        )
        after = self._run(perturbed)

        self.assertEqual(baseline["best_model_name"], after["best_model_name"])
        self.assertEqual(baseline["cv_auroc"], after["cv_auroc"])
        self.assertEqual(
            baseline["decision_threshold"], after["decision_threshold"]
        )
        np.testing.assert_allclose(baseline["prob"], after["prob"])

    def test_threshold_is_the_training_prevalence(self):
        result = self._run(self.frame)
        train_y = self.frame.loc[result["train_index"], "y"]
        self.assertAlmostEqual(
            result["decision_threshold"], float(train_y.mean()), places=12
        )

    def test_scaler_is_fitted_inside_the_model_pipeline(self):
        from src.pipeline import _build_models
        models = _build_models(42)
        for name in ("logistic_regression", "l1_logistic_regression"):
            self.assertEqual(models[name].steps[0][0], "scaler")


if __name__ == "__main__":
    unittest.main()
