"""Synthetic tests for the time-bounded JAMA production scope."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from src import adjustments
from src import extended_analyses as ea
from src import metrics


ROOT = Path(__file__).resolve().parents[1]


class _WarnedFit:
    params = pd.Series({"const": 0.0, "sex": 0.2, "moderator": 0.1,
                        "sex_x_moderator": np.inf})
    pvalues = pd.Series({"sex_x_moderator": np.nan})
    mle_retvals = {"converged": True}

    def conf_int(self):
        return pd.DataFrame(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [np.nan, np.nan]],
            index=self.params.index,
        )


class _WarnedLogit:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        warnings.warn("Hessian inversion failed", RuntimeWarning)
        return _WarnedFit()


class JamaScopeTests(unittest.TestCase):
    @staticmethod
    def _reporting_source_module():
        script_path = ROOT / "scripts" / "build_jama_reporting_sources.py"
        spec = importlib.util.spec_from_file_location(
            "jama_reporting_sources", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _write_figure1_inputs(root, module):
        for index, (_, relative, _) in enumerate(
            module.SEX_FIGURE1_SOURCES, 1
        ):
            gap = -0.01 * index
            low = gap - 0.02
            high = gap + (0.005 if index % 2 else 0.03)
            pd.DataFrame([{
                "dataset": f"sex_{index}",
                "metric": "sensitivity",
                "raw_gap": gap,
                "raw_ci_low": low,
                "raw_ci_high": high,
            }]).to_csv(root / relative, index=False)

        age_dir = root / "extended" / "age_decomposition"
        race_dir = root / "extended" / "race_decomposition"
        age_dir.mkdir(parents=True)
        race_dir.mkdir(parents=True)
        age_rows = []
        for index, dataset in enumerate(module.AGE_FIGURE1_DATASETS, 1):
            gap = -0.1 * index
            age_rows.append({
                "dataset": dataset,
                "group_a": f"young_{index}",
                "group_b": f"old_{index}",
                "sensitivity_gap": gap,
                "sensitivity_ci_low": gap - 0.04,
                "sensitivity_ci_high": gap + 0.04,
                "exploratory": dataset.startswith("NHANES"),
            })
        pd.DataFrame(age_rows).to_csv(
            age_dir / "age_raw_gaps.csv", index=False
        )

        race_rows = []
        for index, (dataset, comparison) in enumerate(
            module.RACE_FIGURE1_COMPARISONS, 1
        ):
            group_a, group_b = comparison.split("_vs_")
            gap = 0.01 * index
            race_rows.append({
                "dataset": dataset,
                "comparison": comparison,
                "group_a": group_a,
                "group_b": group_b,
                "sensitivity_gap": gap,
                "sensitivity_ci_low": gap - 0.03,
                "sensitivity_ci_high": gap + 0.03,
                "exploratory": dataset.startswith("NHANES"),
                "is_secondary": False,
            })
        pd.DataFrame(race_rows).to_csv(
            race_dir / "race_raw_gaps.csv", index=False
        )

    def test_nonfinite_interaction_is_explicitly_nonestimable(self):
        y = np.array([1] * 40 + [0] * 40)
        pred = np.array(([0, 1] * 20) + ([0, 1] * 20))
        group = np.array((["A", "B"] * 20) + (["A", "B"] * 20))
        moderator = np.linspace(-2, 2, 80)
        with patch.object(ea.sm, "Logit", _WarnedLogit):
            result = ea.subgroup_covariate_interaction_test(
                y, pred, group, moderator, "A", "B", error_type="fn"
            )
        self.assertEqual(result["status"], "nonestimable")
        self.assertIn("Hessian inversion failed", result["warning"])
        self.assertIn("nonfinite", result["note"])
        for field in (
            "interaction_logOR", "interaction_OR", "interaction_OR_ci_low",
            "interaction_OR_ci_high", "interaction_p", "main_effect_sex_OR",
            "main_effect_moderator_OR",
        ):
            self.assertTrue(np.isnan(result[field]), field)

    def test_insufficient_interaction_has_complete_nonestimable_schema(self):
        result = ea.subgroup_covariate_interaction_test(
            [1, 1, 0, 0], [0, 1, 0, 1], ["A", "B", "A", "B"],
            [0.1, 0.2, 0.3, 0.4], "A", "B",
        )
        self.assertEqual(result["status"], "nonestimable")
        self.assertEqual(result["warning"], "")
        self.assertEqual(result["error"], "")
        self.assertTrue(np.isnan(result["interaction_p"]))

    def test_group_metric_confidence_intervals_have_documented_schema(self):
        values = metrics.threshold_metrics(
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 1]),
        )
        intervals = metrics.threshold_metric_confidence_intervals(values)
        self.assertEqual(
            set(intervals), {"prevalence", "sensitivity", "specificity", "ppv", "npv"}
        )
        for record in intervals.values():
            self.assertEqual(record["method"], "wilson")
            self.assertLessEqual(record["ci_low"], record["point"])
            self.assertGreaterEqual(record["ci_high"], record["point"])

    def test_zero_bootstrap_mode_returns_points_without_intervals(self):
        result = adjustments.bootstrap_gap_ci(
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
            np.linspace(0.1, 0.8, 8),
            np.array([0, 0, 1, 0, 1, 1, 1, 0]),
            np.array([True, True, True, True, False, False, False, False]),
            np.array([False, False, False, False, True, True, True, True]),
            "sensitivity", n_boot=0,
        )
        self.assertEqual(result["boot"].size, 0)
        self.assertTrue(np.isnan(result["ci_low"]))
        self.assertTrue(np.isnan(result["ci_high"]))

    def test_reporting_source_builder_uses_only_associated_primary_records(self):
        module = self._reporting_source_module()
        group = {
            "label": "A", "value": "A",
            "counts": {"n": 20, "n_pos": 5, "n_neg": 15,
                       "tp": 4, "fp": 2, "tn": 13, "fn": 1},
            "metrics_with_confidence_intervals": {
                "sensitivity": {"point": 0.8, "ci_low": 0.4, "ci_high": 0.96,
                                "method": "wilson", "numerator": 4, "denominator": 5}
            },
        }
        record = {
            "command_id": "sex_cdc",
            "analysis": {"dataset": "Synthetic", "protected_attribute": "sex",
                         "comparison": "A vs B"},
            "reporting": {"group_a": group, "group_b": {**group, "label": "B", "value": "B"}},
            "model_selection": {"selected_family": "logistic_regression"},
            "final_evaluation": {"test_auroc": 0.75},
            "threshold": {"value": 0.2},
        }
        denominators, table = module.build_reporting_tables([record])
        self.assertEqual(len(denominators), 2)
        self.assertEqual(len(table), 2)
        self.assertEqual(set(table["dataset"]), {"Synthetic"})
        self.assertEqual(set(table["analysis_role"]), {"primary"})

    def test_figure1_source_has_exact_complete_13_point_contract(self):
        module = self._reporting_source_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_figure1_inputs(root, module)
            figure = module.build_figure1_source(root)
        self.assertEqual(tuple(figure.columns), module.FIGURE1_COLUMNS)
        self.assertEqual(len(figure), 13)
        self.assertFalse(
            figure[["sensitivity_gap", "ci_low", "ci_high"]].isna().any().any()
        )
        self.assertEqual(list(figure["display_order"]), list(range(1, 14)))
        self.assertEqual(
            figure.groupby("protected_attribute").size().to_dict(),
            {"sex": 5, "age": 5, "race/ethnicity": 3},
        )
        self.assertEqual(
            figure["analysis_status"].value_counts().to_dict(),
            {"primary": 10, "exploratory": 3},
        )

    def test_figure1_values_trace_to_sources_and_match_caption_conventions(self):
        module = self._reporting_source_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_figure1_inputs(root, module)
            figure = module.build_figure1_source(root)
            for _, point in figure.iterrows():
                relative = point["source_result_path"].removeprefix("results/")
                source = pd.read_csv(root / relative)
                if point["protected_attribute"] == "sex":
                    value = source[source["metric"] == "sensitivity"].iloc[0]
                    fields = ("raw_gap", "raw_ci_low", "raw_ci_high")
                    self.assertEqual(point["comparison_label"], "Male minus Female")
                    self.assertEqual(
                        (point["group_a"], point["group_b"]),
                        ("Male", "Female"),
                    )
                elif point["protected_attribute"] == "age":
                    value = source[source["dataset"] == point["dataset"]].iloc[0]
                    fields = (
                        "sensitivity_gap",
                        "sensitivity_ci_low",
                        "sensitivity_ci_high",
                    )
                    self.assertEqual(
                        point["comparison_label"], "Youngest minus Oldest"
                    )
                else:
                    value = source[
                        (source["dataset"] == point["dataset"])
                        & (source["group_a"] == point["group_a"])
                        & (source["group_b"] == point["group_b"])
                    ].iloc[0]
                    fields = (
                        "sensitivity_gap",
                        "sensitivity_ci_low",
                        "sensitivity_ci_high",
                    )
                    self.assertEqual(
                        point["comparison_label"],
                        f"{point['group_a']} minus {point['group_b']}",
                    )
                self.assertEqual(point["sensitivity_gap"], value[fields[0]])
                self.assertEqual(point["ci_low"], value[fields[1]])
                self.assertEqual(point["ci_high"], value[fields[2]])

        race = figure[figure["protected_attribute"] == "race/ethnicity"]
        self.assertFalse(race["dataset"].str.contains("CDC|CCHS").any())
        self.assertFalse(
            race["comparison_label"].str.contains(
                "Hispanic|Multiracial", case=False, regex=True
            ).any()
        )

    def test_figure1_validator_rejects_a_missing_plotted_value(self):
        module = self._reporting_source_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_figure1_inputs(root, module)
            figure = module.build_figure1_source(root)
        figure.loc[0, "ci_low"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing required"):
            module.validate_figure1_source(figure)

    def test_eight_thread_runtime_profile_is_set_before_runtime_import(self):
        environment = dict(os.environ)
        environment["MEDICAL_FAIRNESS_N_JOBS"] = "8"
        completed = subprocess.run(
            [sys.executable, "-B", "-c",
             "import json; from src import runtime; "
             "print(json.dumps({'n':runtime.DETERMINISTIC_N_JOBS,'e':runtime.THREAD_ENVIRONMENT}))"],
            cwd=ROOT, env=environment, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["n"], 8)
        self.assertEqual(set(payload["e"].values()), {"8"})

    def test_eight_thread_preflight_reports_absolute_only_diagnostic_tolerance(self):
        environment = dict(os.environ)
        environment["MEDICAL_FAIRNESS_N_JOBS"] = "8"
        for name in (
            "BLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        ):
            environment[name] = "8"
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "src.reproducibility_preflight"],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["rtol"], 0.0)
        self.assertEqual(payload["continuous_diagnostic_atol"], 1e-5)


if __name__ == "__main__":
    unittest.main()
