"""Fast synthetic producer-consumer contract checks for production preflight.

This module configures deterministic native thread limits before importing any
numerical library. It uses synthetic arrays only and writes only to a temporary
directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from .production import configure_deterministic_environment


configure_deterministic_environment()
os.environ.setdefault("MPLBACKEND", "Agg")


def run_output_contract_preflight() -> tuple[str, ...]:
    """Exercise the threshold-sweep producer and its production plot consumer."""
    import numpy as np
    import pandas as pd

    from . import extended_analyses as ea

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    probabilities = np.array([0.05, 0.92, 0.35, 0.68, 0.15, 0.81, 0.55, 0.45])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    sweep = ea.threshold_sweep(
        y_true,
        probabilities,
        groups,
        group_a_val="A",
        group_b_val="B",
        n_steps=7,
    )
    ea.validate_threshold_sweep_schema(sweep)

    if not np.allclose(
        sweep["predicted_positive_rate_gap"],
        sweep["predicted_positive_rate_A"]
        - sweep["predicted_positive_rate_B"],
        equal_nan=True,
    ):
        raise AssertionError(
            "predicted_positive_rate_gap does not equal group A minus group B"
        )

    # No alias is currently part of this schema. If one is introduced later,
    # this assertion makes its compatibility relationship explicit.
    if "ppr_gap" in sweep.columns and not sweep["ppr_gap"].equals(
        sweep["predicted_positive_rate_gap"]
    ):
        raise AssertionError("ppr_gap alias differs from predicted_positive_rate_gap")

    persisted = sweep.copy()
    persisted.insert(0, "dataset", "synthetic")
    if tuple(persisted.columns) != ea.THRESHOLD_SWEEP_CSV_COLUMNS:
        raise AssertionError("Persisted threshold-sweep CSV schema drifted")

    with tempfile.TemporaryDirectory(prefix="medical_fairness_contract_") as temp_dir:
        output_path = Path(temp_dir) / "threshold_sweep.png"
        ea.save_threshold_sweep_plot(
            sweep,
            operating_threshold=0.5,
            dataset_name="Synthetic contract",
            output_path=output_path,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AssertionError("Threshold-sweep plot consumer produced no PNG")

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
        summary_path = Path(temp_dir) / "calibration_method_robustness_summary.csv"
        summary.to_csv(summary_path, index=False)
        round_trip = pd.read_csv(summary_path)
        if tuple(round_trip.columns) != ea.CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS:
            raise AssertionError("Calibration summary CSV did not round-trip cleanly")
        if not round_trip.columns.is_unique:
            raise AssertionError("Calibration summary CSV columns are not unique")

    return ea.THRESHOLD_SWEEP_COLUMNS


def main() -> int:
    columns = run_output_contract_preflight()
    print("preflight_output_contracts: passed")
    print(",".join(columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
