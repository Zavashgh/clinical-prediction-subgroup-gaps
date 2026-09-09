"""Synthetic integrity tests that never read or write production results."""

import json
import os
from pathlib import Path
import re
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import adjustments
from src import datasets
from src import pipeline
from src import runtime
from src.evaluation_splits import (
    assert_no_training_overlap,
    isolated_nonprimary_holdout_indices,
)


class _ToyEstimator:
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        values = np.asarray(X, dtype=float)[:, 0]
        span = np.ptp(values)
        scaled = (values - values.min()) / span if span else np.full(len(values), 0.5)
        positive = np.clip(0.15 + 0.70 * scaled, 0.01, 0.99)
        return np.column_stack([1.0 - positive, positive])


class _ToyCalibrator:
    def __init__(self, estimator, fit_indices):
        self.estimator = estimator
        self.fit_indices = fit_indices

    def fit(self, X, y):
        self.fit_indices.append(set(X.index))
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class IntegrityProtocolTests(unittest.TestCase):
    def test_family_selection_and_calibration_receive_training_rows_only(self):
        n = 120
        indices = pd.Index(np.arange(1000, 1000 + n))
        frame = pd.DataFrame(
            {
                "feature": np.linspace(-2.0, 2.0, n),
                "outcome": np.tile([0, 1, 0, 1], n // 4),
                "group": np.tile(["A", "A", "B", "B"], n // 4),
            },
            index=indices,
        )
        cv_indices = []
        calibration_fit_indices = []

        def fake_cross_val_score(model, X, y, cv, scoring, n_jobs):
            cv_indices.append(set(X.index))
            return np.array([0.70, 0.71, 0.69, 0.72, 0.70])

        def fake_calibrator(estimator, cv=5):
            return _ToyCalibrator(estimator, calibration_fit_indices)

        with (
            patch.object(pipeline, "_build_models", return_value={"toy": _ToyEstimator()}),
            patch.object(pipeline, "cross_val_score", side_effect=fake_cross_val_score),
            patch.object(pipeline, "_make_calibrated_classifier", side_effect=fake_calibrator),
        ):
            result = pipeline.run_fairness_analysis(
                frame,
                feature_cols=["feature"],
                target_col="outcome",
                group_col="group",
                group_a_value="A",
                group_b_value="B",
                threshold="prevalence",
                random_state=42,
                n_boot=5,
                run_calibration_adjustment=False,
                run_case_mix=False,
                verbose=False,
            )

        train_indices = set(result["train_index"])
        test_indices = set(result["test_index"])
        self.assertFalse(train_indices & test_indices)
        self.assertEqual(cv_indices, [train_indices])
        self.assertEqual(calibration_fit_indices, [train_indices])

    def test_controlled_failure_when_every_family_is_invalid(self):
        scores = {"failed": np.nan, "chance": 0.5, "worse": 0.49, "infinite": np.inf}
        with self.assertRaisesRegex(RuntimeError, "Per-family diagnostics") as error:
            pipeline._select_best_model(scores, {"failed": "ValueError: synthetic failure"})
        message = str(error.exception)
        for name in scores:
            self.assertIn(name, message)

    def test_required_family_cv_failure_aborts_before_selection(self):
        with self.assertRaisesRegex(RuntimeError, "not fully available") as error:
            pipeline._require_complete_cv_evaluation(
                ["eligible", "failed"],
                {"eligible": 0.80, "failed": np.nan},
                {"failed": "ValueError: synthetic fit failure"},
            )
        self.assertIn("failed [failed: ValueError", str(error.exception))

    def test_equal_sensitivity_threshold_has_no_off_by_one(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_prob = np.array([0.90, 0.80, 0.70, 0.60, 0.95, 0.55, 0.40, 0.10])
        groups = np.array(["A"] * len(y_true))
        threshold = adjustments.equal_sensitivity_thresholds(
            y_true, y_prob, groups, ["A"], target_sensitivity=0.50
        )["A"]
        sensitivity = ((y_prob >= threshold) & (y_true == 1)).sum() / y_true.sum()
        self.assertEqual(threshold, 0.80)
        self.assertEqual(sensitivity, 0.50)

    def test_signed_attenuation_represents_reversal_above_100_percent(self):
        self.assertAlmostEqual(adjustments.attenuation_pct(-0.10, -0.04), 60.0)
        self.assertAlmostEqual(adjustments.attenuation_pct(-0.10, 0.02), 120.0)
        self.assertAlmostEqual(adjustments.attenuation_pct(0.10, -0.02), 120.0)
        self.assertLess(adjustments.attenuation_pct(0.10, 0.15), 0.0)

    def test_calibration_protocol_is_explicit_fold_ensemble(self):
        calibrator = pipeline._make_calibrated_classifier(_ToyEstimator(), cv=5)
        self.assertEqual(calibrator.method, "isotonic")
        self.assertEqual(calibrator.cv, 5)
        self.assertEqual(calibrator.n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertIs(calibrator.ensemble, True)
        self.assertEqual(
            pipeline.CALIBRATION_PROTOCOL,
            "five_fold_isotonic_fold_ensemble",
        )

    def test_required_candidate_pool_is_exactly_six_and_fixed_threaded(self):
        models = pipeline._build_models(random_state=42)
        self.assertEqual(tuple(models), pipeline.REQUIRED_MODEL_FAMILIES)
        self.assertEqual(len(models), 6)
        self.assertEqual(models["random_forest"].n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertEqual(models["xgboost"].n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertEqual(models["xgboost"].device, "cpu")
        ensemble = models["calibrated_ensemble"]
        self.assertEqual(ensemble.n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertEqual(ensemble.named_estimators["rf"].n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertEqual(ensemble.named_estimators["xgb"].n_jobs, runtime.DETERMINISTIC_N_JOBS)
        self.assertEqual(ensemble.named_estimators["xgb"].device, "cpu")

    def test_missing_xgboost_fails_closed_with_controlled_diagnostic(self):
        import_error = ImportError("synthetic unavailable dependency")
        with (
            patch.object(pipeline, "_HAS_XGB", False),
            patch.object(pipeline, "_XGB_IMPORT_ERROR", import_error),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "required six-family candidate pool"
            ) as error:
                pipeline._build_models(random_state=42)
        self.assertIn("No reduced four-family fallback", str(error.exception))

    def test_primary_threshold_is_training_prevalence_and_numeric_is_opt_in(self):
        frame = pd.DataFrame(
            {
                "feature": np.linspace(-2.0, 2.0, 120),
                "outcome": np.tile([0, 1, 0, 1], 30),
                "group": np.tile(["A", "A", "B", "B"], 30),
            }
        )

        def fake_cross_val_score(model, X, y, cv, scoring, n_jobs):
            return np.full(5, 0.70)

        def fake_calibrator(estimator, cv=5):
            return _ToyCalibrator(estimator, [])

        common = dict(
            df=frame,
            feature_cols=["feature"],
            target_col="outcome",
            group_col="group",
            group_a_value="A",
            group_b_value="B",
            n_boot=2,
            run_calibration_adjustment=False,
            run_case_mix=False,
            verbose=False,
        )
        with (
            patch.object(pipeline, "_build_models", return_value={"toy": _ToyEstimator()}),
            patch.object(pipeline, "cross_val_score", side_effect=fake_cross_val_score),
            patch.object(pipeline, "_make_calibrated_classifier", side_effect=fake_calibrator),
        ):
            primary = pipeline.run_fairness_analysis(**common)
            self.assertEqual(primary["threshold_protocol"], "training_outcome_prevalence")
            self.assertAlmostEqual(
                primary["decision_threshold"],
                frame.loc[primary["train_index"], "outcome"].mean(),
            )
            with self.assertRaisesRegex(ValueError, "Primary analyses"):
                pipeline.run_fairness_analysis(**common, threshold=0.5)
            nonprimary = pipeline.run_fairness_analysis(
                **common, threshold=0.5, allow_nonprimary_threshold=True
            )
            self.assertEqual(nonprimary["threshold_protocol"], "explicit_numeric_nonprimary")

    def test_thread_configuration_matches_fixed_profile(self):
        effective = runtime.configure_deterministic_environment()
        self.assertEqual(effective, runtime.THREAD_ENVIRONMENT)
        expected = str(runtime.DETERMINISTIC_N_JOBS)
        self.assertTrue(all(value == expected for value in effective.values()))
        provenance = runtime.runtime_provenance()
        self.assertEqual(provenance["n_jobs"], runtime.DETERMINISTIC_N_JOBS)
        self.assertTrue(
            all(pool["num_threads"] == runtime.DETERMINISTIC_N_JOBS
                for pool in provenance["loaded_threadpools"])
        )

    def test_l1_ratio_equivalent_to_legacy_l1_predictions_and_cv(self):
        X, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            random_state=17,
        )
        legacy = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", C=0.5,
                max_iter=2000, random_state=42, n_jobs=1,
            )),
        ])
        supported = pipeline._build_models(42)["l1_logistic_regression"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            legacy.fit(X, y)
            supported.fit(X, y)
            np.testing.assert_allclose(
                legacy.predict_proba(X), supported.predict_proba(X),
                rtol=0.0, atol=1e-12,
            )
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            legacy_scores = cross_val_score(
                legacy, X, y, cv=folds, scoring="roc_auc", n_jobs=1
            )
            supported_scores = cross_val_score(
                supported, X, y, cv=folds, scoring="roc_auc", n_jobs=1
            )
        np.testing.assert_allclose(legacy_scores, supported_scores, rtol=0.0, atol=1e-12)

    def test_selected_family_is_deterministic_under_single_thread_runtime(self):
        X, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            class_sep=1.2,
            random_state=23,
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def select_once():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                fold_scores = {
                    name: cross_val_score(
                        model, X, y, cv=folds, scoring="roc_auc", n_jobs=1
                    )
                    for name, model in pipeline._build_models(42).items()
                }
            means = {name: float(scores.mean()) for name, scores in fold_scores.items()}
            return pipeline._select_best_model(means), fold_scores

        first_name, first_scores = select_once()
        second_name, second_scores = select_once()
        self.assertEqual(first_name, second_name)
        self.assertEqual(tuple(first_scores), pipeline.REQUIRED_MODEL_FAMILIES)
        for name in pipeline.REQUIRED_MODEL_FAMILIES:
            np.testing.assert_array_equal(first_scores[name], second_scores[name])

    def test_dataset_paths_do_not_depend_on_callers_working_directory(self):
        filenames = (
            "cdc_diabetes.csv",
            "diabetes_130_hospitals.csv",
            "brfss2022_subset.csv",
            "nhanes_2017_2018.csv",
            "cchs_2019_2020_subset.csv",
        )
        expected_data_dir = Path(datasets.__file__).resolve().parents[1] / "data"
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                for filename in filenames:
                    self.assertEqual(
                        datasets._resolve_data_path(None, filename),
                        expected_data_dir / filename,
                    )
                    self.assertEqual(
                        datasets._resolve_data_path(f"../data/{filename}", filename),
                        expected_data_dir / filename,
                    )
            finally:
                os.chdir(original_cwd)

    def test_auxiliary_holdout_is_deterministic_unique_and_training_isolated(self):
        frame = pd.DataFrame(
            {
                "outcome": np.tile([0, 1], 40),
                "group": np.repeat(["primary_a", "primary_b", "extra_c", "extra_d"], 20),
            },
            index=np.arange(5000, 5080),
        )
        training_index = frame.index[frame["group"].isin(["primary_a", "primary_b"])]
        first = isolated_nonprimary_holdout_indices(
            frame, "outcome", "group", ["extra_c", "extra_d"], random_state=42
        )
        second = isolated_nonprimary_holdout_indices(
            frame, "outcome", "group", ["extra_c", "extra_d"], random_state=42
        )
        self.assertTrue(first.equals(second))
        self.assertFalse(first.has_duplicates)
        self.assertEqual(set(frame.loc[first, "group"]), {"extra_c", "extra_d"})
        self.assertTrue(assert_no_training_overlap(training_index, first, "synthetic"))
        with self.assertRaisesRegex(RuntimeError, "overlap model training"):
            assert_no_training_overlap(training_index, training_index[:1], "synthetic")

    def test_notebook_production_sources_use_rooted_paths_and_primary_threshold(self):
        root = Path(pipeline.__file__).resolve().parents[1]
        notebooks = sorted((root / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 6)
        code_by_name = {}
        for notebook in notebooks:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in payload["cells"]
                if cell.get("cell_type") == "code"
            )
            code_by_name[notebook.name] = code
            self.assertNotIn("../data", code)
            self.assertNotIn("../results", code)
            self.assertNotIn('sys.path.insert(0, "..")', code)
            self.assertIn("RESULTS_DIR = ROOT / \"results\"", code)
        cdc_source = code_by_name["01_cdc_diabetes.ipynb"]
        self.assertIn('threshold="prevalence"', cdc_source)
        self.assertNotIn("threshold=0.5", cdc_source)

    def test_robustness_baselines_do_not_restore_cdc_fixed_threshold(self):
        root = Path(pipeline.__file__).resolve().parents[1]
        scripts = sorted((root / "extended_robustness_scripts").glob("*.py"))
        self.assertGreater(len(scripts), 0)
        fixed_config = re.compile(r'"threshold"\s*:\s*0\.5')
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            self.assertIsNone(fixed_config.search(source), script.name)

    def test_pipeline_provenance_capture_is_strict_json_and_complete(self):
        cv_details = {
            name: {
                "status": "eligible",
                "fold_scores": [0.70, 0.71, 0.72, 0.73, 0.74],
                "mean": 0.72,
                "sd": 0.0158,
                "error": None,
            }
            for name in pipeline.REQUIRED_MODEL_FAMILIES
        }
        result = {
            "runtime_provenance": {
                "n_jobs": runtime.DETERMINISTIC_N_JOBS,
                "thread_environment": dict(runtime.THREAD_ENVIRONMENT),
                "loaded_threadpools": [],
            },
            "candidate_model_families": list(pipeline.REQUIRED_MODEL_FAMILIES),
            "best_model_name": "logistic_regression",
            "cv_auroc_details": cv_details,
            "calibration_protocol": pipeline.CALIBRATION_PROTOCOL,
            "selection_cv_protocol": "stratified_kfold_5",
            "split_protocol": pipeline.SPLIT_PROTOCOL_ROW,
            "bootstrap_protocol": pipeline.BOOTSTRAP_PROTOCOL_ROW,
            "split_diagnostics": {
                "split_protocol": pipeline.SPLIT_PROTOCOL_ROW,
                "n_train": 84,
                "n_test": 36,
                "n_train_events": 14,
                "n_test_events": 6,
                "outcome_stratified": True,
                "grouped_by_cluster": False,
            },
            "preprocessing_schema": {
                "feature_order_declared": ["x1"],
                "numeric_cols": ["x1"],
                "imputed_cols": [],
                "training_medians": {},
                "categorical_sources": [],
                "feature_names_in_order": ["x1"],
                "n_features": 1,
                "unseen_categories_by_partition": {},
            },
            "threshold_protocol": pipeline.PRIMARY_THRESHOLD_PROTOCOL,
            "threshold_request": "prevalence",
            "decision_threshold": 0.2,
            "final_test_auroc": 0.75,
            "n_test": 36,
            "case_mix_inference_protocol": pipeline.CASE_MIX_INFERENCE_PROTOCOL,
            "subgroup_metrics": {
                "A": {"n": 18, "n_pos": 6, "n_neg": 12, "tp": 5, "fp": 2, "tn": 10, "fn": 1},
                "B": {"n": 18, "n_pos": 6, "n_neg": 12, "tp": 4, "fp": 3, "tn": 9, "fn": 2},
            },
            "subgroup_metric_cis": {
                "A": {"sensitivity": {"point": 5 / 6, "ci_low": 0.4, "ci_high": 0.97,
                                        "method": "wilson", "numerator": 5, "denominator": 6}},
                "B": {"sensitivity": {"point": 4 / 6, "ci_low": 0.3, "ci_high": 0.9,
                                        "method": "wilson", "numerator": 4, "denominator": 6}},
            },
        }
        inputs = [{"path": "data/example.csv", "sha256": "a" * 64, "size_bytes": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "pipeline.jsonl"
            environment = {
                "MEDICAL_FAIRNESS_PROVENANCE_JSONL": str(destination),
                "MEDICAL_FAIRNESS_INPUT_HASHES_JSON": json.dumps(inputs),
                "MEDICAL_FAIRNESS_COMMAND_ID": "synthetic_test",
                "MEDICAL_FAIRNESS_GIT_COMMIT": "b" * 40,
                "MEDICAL_FAIRNESS_GIT_BRANCH": "main",
                "MEDICAL_FAIRNESS_SOURCE_DIRTY": "0",
            }
            with patch.dict(os.environ, environment, clear=False):
                pipeline._persist_pipeline_provenance(
                    result,
                    target_col="outcome",
                    group_col="group",
                    group_a_value="A",
                    group_b_value="B",
                    label_a="Group A",
                    label_b="Group B",
                    random_state=42,
                    n_rows=120,
                    n_features=1,
                    analysis_label="Synthetic dataset",
                )
            records = [json.loads(line) for line in destination.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["git"]["commit"], "b" * 40)
        self.assertEqual(record["inputs"], inputs)
        self.assertEqual(record["model_selection"]["selected_family"], "logistic_regression")
        self.assertEqual(
            tuple(record["model_selection"]["candidate_families"]),
            pipeline.REQUIRED_MODEL_FAMILIES,
        )
        self.assertEqual(record["threshold"]["derivation_partition"], "training_set")
        self.assertIs(record["calibration"]["ensemble"], True)
        self.assertEqual(record["analysis"]["dataset"], "Synthetic dataset")
        self.assertEqual(record["reporting"]["group_a"]["counts"]["tp"], 5)

if __name__ == "__main__":
    unittest.main()
