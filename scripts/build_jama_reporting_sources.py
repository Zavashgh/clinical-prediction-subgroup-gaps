"""Build JAMA table/flow source CSVs from finalized analytical provenance."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import configure_deterministic_environment, enforce_loaded_threadpool_limits

configure_deterministic_environment()

import pandas as pd

_THREAD_LIMITER = enforce_loaded_threadpool_limits()

from src import datasets


OUTPUT_ROOT = Path(os.environ.get("MEDICAL_FAIRNESS_RESULTS_DIR", ROOT / "results"))
OUT_DIR = OUTPUT_ROOT / "jama_reporting"

PRIMARY_COMMANDS = {
    "sex_cdc", "sex_diabetes130", "sex_brfss", "sex_nhanes", "sex_cchs",
    "age_decomposition", "race_decomposition",
}
PRIMARY_RACE_COMPARISONS = {
    "AfricanAmerican vs Caucasian", "Black vs White",
}

RAW_DATASETS = (
    ("CDC Diabetes (BRFSS 2015)", datasets.load_cdc_diabetes,
     "data/cdc_diabetes.csv"),
    ("Diabetes-130 (readmission)", datasets.load_diabetes130,
     "data/diabetes_130_hospitals.csv"),
    ("BRFSS 2022 (heart disease)", datasets.load_brfss,
     "data/brfss2022_subset.csv"),
    ("NHANES 2017-18 (diabetes)", datasets.load_nhanes,
     "data/nhanes_2017_2018.csv"),
    ("CCHS 2019-20 (diabetes)", datasets.load_cchs,
     "data/cchs_2019_2020_subset.csv"),
)

FIGURE1_COLUMNS = (
    "dataset",
    "protected_attribute",
    "comparison_label",
    "group_a",
    "group_b",
    "sensitivity_gap",
    "ci_low",
    "ci_high",
    "significance",
    "analysis_status",
    "display_order",
    "source_result_path",
)

SEX_FIGURE1_SOURCES = (
    ("CDC Diabetes (BRFSS 2015)", "01_cdc_diabetes_summary.csv", False),
    ("Diabetes-130 (readmission)", "02_diabetes130_summary.csv", False),
    ("BRFSS 2022 (heart disease)", "03_brfss_summary.csv", False),
    ("NHANES 2017-18 (diabetes)", "04_nhanes_summary.csv", True),
    ("CCHS 2019-20 (diabetes)", "05_cchs_summary.csv", False),
)

AGE_FIGURE1_DATASETS = (
    "CDC Diabetes (BRFSS 2015)",
    "Diabetes-130 (readmission)",
    "BRFSS 2022 (heart disease)",
    "NHANES 2017-18 (diabetes)",
    "CCHS 2019-20 (diabetes)",
)

RACE_FIGURE1_COMPARISONS = (
    ("Diabetes-130 (readmission)", "AfricanAmerican_vs_Caucasian"),
    ("BRFSS 2022 (heart disease)", "Black_vs_White"),
    ("NHANES 2017-18 (diabetes)", "Black_vs_White"),
)


def _records(provenance):
    records = []
    for line in provenance.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("record_type") == "pipeline_analysis":
                records.append(record)
    return records


def _analysis_role(record):
    command = record.get("command_id")
    comparison = record["analysis"]["comparison"]
    if command not in PRIMARY_COMMANDS:
        return "supplement_recomputation"
    if command == "race_decomposition" and comparison not in PRIMARY_RACE_COMPARISONS:
        return "prespecified_secondary_supplement"
    return "primary"


def build_reporting_tables(records):
    denominator_rows, metric_rows = [], []
    for record in records:
        role = _analysis_role(record)
        if role == "supplement_recomputation":
            continue
        analysis = record["analysis"]
        for group_key in ("group_a", "group_b"):
            group = record["reporting"][group_key]
            counts = group["counts"]
            base = {
                "dataset": analysis.get("dataset") or record["command_id"],
                "command_id": record["command_id"],
                "analysis_role": role,
                "protected_attribute": analysis["protected_attribute"],
                "comparison": analysis["comparison"],
                "group": group["label"],
                "group_value": group["value"],
            }
            denominator_rows.append({**base, **counts})
            for metric, values in group["metrics_with_confidence_intervals"].items():
                metric_rows.append({
                    **base,
                    "metric": metric,
                    "estimate": values["point"],
                    "ci_low": values["ci_low"],
                    "ci_high": values["ci_high"],
                    "ci_method": values["method"],
                    "numerator": values["numerator"],
                    "denominator": values["denominator"],
                    "selected_family": record["model_selection"]["selected_family"],
                    "test_auroc": record["final_evaluation"]["test_auroc"],
                    "threshold": record["threshold"]["value"],
                })
    return pd.DataFrame(denominator_rows), pd.DataFrame(metric_rows)


def build_missing_data_flow():
    rows = []
    for name, loader, relative in RAW_DATASETS:
        path = ROOT / relative
        raw = pd.read_csv(path)
        loaded = loader(path=str(path))
        cleaned = loaded["df"]
        feature_cols = loaded["feature_cols"]
        rows.append({
            "dataset": name,
            "raw_input_path": relative,
            "raw_rows": len(raw),
            "analysis_rows_after_loader": len(cleaned),
            "rows_excluded_by_loader": len(raw) - len(cleaned),
            "raw_missing_cells": int(raw.isna().sum().sum()),
            "analysis_missing_cells": int(cleaned.isna().sum().sum()),
            "predictor_missing_cells_before_imputation": int(
                cleaned[feature_cols].isna().sum().sum()
            ),
            "imputation_timing": "median_imputation_fitted_on_training_rows_only",
            "definition": (
                "descriptive loader-flow counts after fixed recoding and before "
                "training-fitted imputation; no survey weighting"
            ),
        })
    return pd.DataFrame(rows)


def _significance(ci_low, ci_high):
    return (
        "significant"
        if float(ci_low) > 0.0 or float(ci_high) < 0.0
        else "not_significant"
    )


def _analysis_status(exploratory):
    if isinstance(exploratory, str):
        exploratory = exploratory.strip().lower() in {"1", "true", "yes"}
    return "exploratory" if bool(exploratory) else "primary"


def _figure1_row(
    *, dataset, protected_attribute, comparison_label, group_a, group_b,
    estimate, ci_low, ci_high, exploratory, display_order, source_result_path,
):
    return {
        "dataset": dataset,
        "protected_attribute": protected_attribute,
        "comparison_label": comparison_label,
        "group_a": group_a,
        "group_b": group_b,
        "sensitivity_gap": float(estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significance": _significance(ci_low, ci_high),
        "analysis_status": _analysis_status(exploratory),
        "display_order": int(display_order),
        "source_result_path": source_result_path,
    }


def validate_figure1_source(frame):
    """Fail closed unless the JAMA dot-and-whisker source is complete."""
    if tuple(frame.columns) != FIGURE1_COLUMNS:
        raise ValueError(
            "Figure 1 source schema mismatch: "
            f"expected={FIGURE1_COLUMNS!r}; actual={tuple(frame.columns)!r}"
        )
    if len(frame) != 13:
        raise ValueError(f"Figure 1 requires exactly 13 points; found {len(frame)}")
    if frame[list(FIGURE1_COLUMNS)].isna().any().any():
        raise ValueError("Figure 1 contains a missing required plotted field")
    required_text = (
        "dataset", "protected_attribute", "comparison_label", "group_a",
        "group_b", "significance", "analysis_status", "source_result_path",
    )
    if any((frame[column].astype(str).str.strip() == "").any()
           for column in required_text):
        raise ValueError("Figure 1 contains an empty required descriptive field")
    if list(frame["display_order"]) != list(range(1, 14)):
        raise ValueError("Figure 1 display_order must be exactly 1 through 13")
    expected_counts = {"sex": 5, "age": 5, "race/ethnicity": 3}
    if frame.groupby("protected_attribute").size().to_dict() != expected_counts:
        raise ValueError("Figure 1 must contain 5 sex, 5 age, and 3 race points")
    race = frame[frame["protected_attribute"] == "race/ethnicity"]
    if race["dataset"].str.contains("CDC|CCHS", regex=True).any():
        raise ValueError("CDC and CCHS must be absent from Figure 1 race results")
    if race["comparison_label"].str.contains(
        "Hispanic|Multiracial", case=False, regex=True
    ).any():
        raise ValueError("Secondary race comparisons must not appear in Figure 1")
    expected_significance = [
        _significance(low, high)
        for low, high in zip(frame["ci_low"], frame["ci_high"])
    ]
    if list(frame["significance"]) != expected_significance:
        raise ValueError("Figure 1 significance classification does not match its CIs")
    expected_status = [
        "exploratory" if dataset.startswith("NHANES") else "primary"
        for dataset in frame["dataset"]
    ]
    if list(frame["analysis_status"]) != expected_status:
        raise ValueError("Figure 1 exploratory/primary status is inconsistent")
    return frame


def build_figure1_source(output_root=OUTPUT_ROOT):
    """Build the 13-point JAMA sensitivity-gap dot-and-whisker source.

    Each estimate and confidence limit is read from the finalized result CSV
    named in ``source_result_path``. Gaps follow the caption convention group
    A minus group B: male minus female, youngest minus oldest, and the named
    race/ethnicity minority group minus its White/Caucasian reference.
    """
    output_root = Path(output_root)
    rows = []
    display_order = 1

    for dataset, relative, exploratory in SEX_FIGURE1_SOURCES:
        source = output_root / relative
        frame = pd.read_csv(source)
        selected = frame[frame["metric"] == "sensitivity"]
        if len(selected) != 1:
            raise ValueError(f"Expected one sensitivity row in {source}; found {len(selected)}")
        value = selected.iloc[0]
        rows.append(_figure1_row(
            dataset=dataset,
            protected_attribute="sex",
            comparison_label="Male minus Female",
            group_a="Male",
            group_b="Female",
            estimate=value["raw_gap"],
            ci_low=value["raw_ci_low"],
            ci_high=value["raw_ci_high"],
            exploratory=exploratory,
            display_order=display_order,
            source_result_path=f"results/{relative}",
        ))
        display_order += 1

    age_relative = "extended/age_decomposition/age_raw_gaps.csv"
    age = pd.read_csv(output_root / age_relative).set_index("dataset", drop=False)
    for dataset in AGE_FIGURE1_DATASETS:
        if dataset not in age.index:
            raise ValueError(f"Missing Figure 1 age result for {dataset}")
        value = age.loc[dataset]
        rows.append(_figure1_row(
            dataset=dataset,
            protected_attribute="age",
            comparison_label="Youngest minus Oldest",
            group_a=value["group_a"],
            group_b=value["group_b"],
            estimate=value["sensitivity_gap"],
            ci_low=value["sensitivity_ci_low"],
            ci_high=value["sensitivity_ci_high"],
            exploratory=value["exploratory"],
            display_order=display_order,
            source_result_path=f"results/{age_relative}",
        ))
        display_order += 1

    race_relative = "extended/race_decomposition/race_raw_gaps.csv"
    race = pd.read_csv(output_root / race_relative).set_index(
        ["dataset", "comparison"], drop=False
    )
    for dataset, comparison in RACE_FIGURE1_COMPARISONS:
        key = (dataset, comparison)
        if key not in race.index:
            raise ValueError(f"Missing Figure 1 race result for {key}")
        value = race.loc[key]
        rows.append(_figure1_row(
            dataset=dataset,
            protected_attribute="race/ethnicity",
            comparison_label=f"{value['group_a']} minus {value['group_b']}",
            group_a=value["group_a"],
            group_b=value["group_b"],
            estimate=value["sensitivity_gap"],
            ci_low=value["sensitivity_ci_low"],
            ci_high=value["sensitivity_ci_high"],
            exploratory=value["exploratory"],
            display_order=display_order,
            source_result_path=f"results/{race_relative}",
        ))
        display_order += 1

    return validate_figure1_source(pd.DataFrame(rows, columns=FIGURE1_COLUMNS))



# ---------------------------------------------------------------------------
# Consolidated discrimination and calibration tables
# ---------------------------------------------------------------------------
#
# Each comparison writes its own frozen performance file during the analysis.
# These are concatenated here into two reporting sources so the manuscript and
# Supplement read one table each rather than assembling values by hand. No
# statistic is recomputed: every value is read from the frozen files.

PERFORMANCE_COLUMNS = (
    "dataset", "comparison_type", "comparison", "group", "group_value",
    "n", "n_events", "selected_family", "operating_threshold",
    "auroc", "auroc_ci_low", "auroc_ci_high", "auprc",
    "ci_method", "ci_level", "n_boot", "resampling_unit", "resampling_design",
    "n_resampling_units", "n_units_spanning_subgroups",
)

CALIBRATION_COLUMNS = (
    "dataset", "comparison_type", "comparison", "group", "group_value",
    "n", "n_events",
    "brier", "brier_ci_low", "brier_ci_high",
    "calib_intercept", "calib_intercept_ci_low", "calib_intercept_ci_high",
    "calib_slope", "calib_slope_ci_low", "calib_slope_ci_high",
    "calib_status", "calib_slope_valid_replicates",
    "calib_slope_failed_replicates", "calib_replicate_status_counts",
    "calib_n_clipped", "ece",
    "ci_method", "n_boot", "resampling_unit",
)


def _frozen_performance(output_root):
    frozen_dir = Path(output_root) / "frozen"
    files = sorted(frozen_dir.glob("*_performance.csv"))
    if not files:
        raise RuntimeError(
            f"No frozen performance files found under {frozen_dir}; the "
            "analysis commands must run before the reporting sources"
        )
    frame = pd.concat(
        [pd.read_csv(path) for path in files], ignore_index=True
    )
    order = {"sex": 0, "age": 1, "race": 2, "race_secondary": 3}
    frame["_order"] = frame["comparison_type"].map(order).fillna(9)
    frame = frame.sort_values(
        ["_order", "dataset", "group"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)
    return frame


def build_performance_tables(output_root):
    """Return (discrimination table, calibration table) from frozen outputs."""
    frame = _frozen_performance(output_root)
    missing = [c for c in (*PERFORMANCE_COLUMNS, *CALIBRATION_COLUMNS)
               if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Frozen performance files lack columns: {missing}")
    return (
        frame.loc[:, list(PERFORMANCE_COLUMNS)].copy(),
        frame.loc[:, list(CALIBRATION_COLUMNS)].copy(),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance_value = os.environ.get("MEDICAL_FAIRNESS_PROVENANCE_JSONL")
    if not provenance_value:
        raise RuntimeError("MEDICAL_FAIRNESS_PROVENANCE_JSONL is required")
    records = _records(Path(provenance_value))
    denominators, metrics = build_reporting_tables(records)
    if denominators.empty or metrics.empty:
        raise RuntimeError("No finalized primary analytical provenance was available")
    denominators.to_csv(OUT_DIR / "table1_source.csv", index=False)
    metrics.to_csv(OUT_DIR / "etable1_subgroup_metrics_source.csv", index=False)
    build_missing_data_flow().to_csv(OUT_DIR / "missing_data_flow.csv", index=False)
    build_figure1_source(OUTPUT_ROOT).to_csv(
        OUT_DIR / "figure1_source_values.csv", index=False
    )
    discrimination, calibration = build_performance_tables(OUTPUT_ROOT)
    discrimination.to_csv(
        OUT_DIR / "etable_model_performance_source.csv", index=False
    )
    calibration.to_csv(
        OUT_DIR / "etable_calibration_source.csv", index=False
    )


if __name__ == "__main__":
    main()
