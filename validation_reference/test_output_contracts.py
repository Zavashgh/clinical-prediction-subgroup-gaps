"""Synthetic regression tests for production output schemas and consumers."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import extended_analyses as ea
from src.output_contracts import run_output_contract_preflight


class OutputContractTests(unittest.TestCase):
    @staticmethod
    def _sweep():
        return ea.threshold_sweep(
            y_true=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            prob=np.array([0.05, 0.92, 0.35, 0.68, 0.15, 0.81, 0.55, 0.45]),
            g_test=np.array(["A", "A", "A", "A", "B", "B", "B", "B"]),
            group_a_val="A",
            group_b_val="B",
            n_steps=7,
        )

    def test_threshold_sweep_exact_documented_schema(self):
        sweep = self._sweep()
        self.assertEqual(tuple(sweep.columns), ea.THRESHOLD_SWEEP_COLUMNS)
        self.assertEqual(
            ea.THRESHOLD_SWEEP_CSV_COLUMNS,
            ("dataset", *ea.THRESHOLD_SWEEP_COLUMNS),
        )
        for column, _ in ea.THRESHOLD_SWEEP_PLOT_COLUMNS:
            self.assertIn(column, sweep.columns)

    def test_predicted_positive_rate_gap_is_canonical(self):
        sweep = self._sweep()
        np.testing.assert_allclose(
            sweep["predicted_positive_rate_gap"],
            sweep["predicted_positive_rate_A"]
            - sweep["predicted_positive_rate_B"],
            equal_nan=True,
        )
        if "ppr_gap" in sweep.columns:
            self.assertTrue(
                sweep["ppr_gap"].equals(sweep["predicted_positive_rate_gap"])
            )
        else:
            self.assertNotIn("ppr_gap", ea.THRESHOLD_SWEEP_COLUMNS)

    def test_actual_production_plot_consumer_completes(self):
        sweep = self._sweep()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "threshold_sweep.png"
            ea.save_threshold_sweep_plot(sweep, 0.5, "Synthetic", output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

        runner = (
            Path(__file__).resolve().parents[1]
            / "extended_robustness_scripts"
            / "run_extended_analyses.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ea.save_threshold_sweep_plot(", runner)
        self.assertNotIn('(\"ppr_gap\",         \"PPR gap', runner)

    def test_consumer_fails_fast_when_schema_drifts(self):
        sweep = self._sweep().drop(columns=["predicted_positive_rate_gap"])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "should_not_exist.png"
            with self.assertRaisesRegex(ValueError, "Threshold-sweep schema mismatch"):
                ea.save_threshold_sweep_plot(sweep, 0.5, "Synthetic", output)
            self.assertFalse(output.exists())

    def test_fast_output_contract_preflight(self):
        self.assertEqual(run_output_contract_preflight(), ea.THRESHOLD_SWEEP_COLUMNS)

    def test_preflight_configures_threads_before_numerical_imports(self):
        source = (
            Path(__file__).resolve().parents[1] / "src" / "output_contracts.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            source.index("configure_deterministic_environment()"),
            source.index("import numpy as np"),
        )

    def test_calibration_summary_has_flat_unique_round_trip_schema(self):
        method_results = pd.DataFrame(
            {
                "dataset": ["one", "one", "two", "two"],
                "sensitivity_gap": [0.1, 0.2, -0.1, -0.2],
                "fnr_gap": [-0.1, -0.2, 0.1, 0.2],
                "ppv_gap_raw": [0.01, 0.02, -0.01, -0.02],
                "ppv_gap_adj": [0.005, 0.01, -0.005, -0.01],
                "ppr_gap": [0.03, 0.04, -0.03, -0.04],
            }
        )
        summary = ea.calibration_method_robustness_summary(method_results)
        self.assertEqual(
            tuple(summary.columns), ea.CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS
        )
        self.assertTrue(summary.columns.is_unique)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            summary.to_csv(path, index=False)
            round_trip = pd.read_csv(path)
        self.assertEqual(
            tuple(round_trip.columns), ea.CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS
        )
        self.assertTrue(round_trip.columns.is_unique)

        runner = (
            Path(__file__).resolve().parents[1]
            / "extended_robustness_scripts"
            / "run_extended_13_20_34.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ea.calibration_method_robustness_summary(df_gap)", runner)
        self.assertIn("index=False", runner)


if __name__ == "__main__":
    unittest.main()
