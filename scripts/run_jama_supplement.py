"""Generate only the sex-analysis outputs retained for the JAMA Supplement."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import configure_deterministic_environment, enforce_loaded_threadpool_limits

configure_deterministic_environment()

import matplotlib
matplotlib.use("Agg")
import pandas as pd

_THREAD_LIMITER = enforce_loaded_threadpool_limits()

from src import datasets as datasets
from src.extended_analyses import (
    covariate_standardized_error_gap,
    save_threshold_sweep_plot,
    threshold_sweep,
)
from src.pipeline import run_fairness_analysis


OUTPUT_ROOT = Path(os.environ.get("MEDICAL_FAIRNESS_RESULTS_DIR", ROOT / "results"))
OUT_DIR = OUTPUT_ROOT / "extended" / "jama_supplement"
FIG_DIR = OUT_DIR / "figures"

CONFIGS = (
    ("CDC Diabetes (BRFSS 2015)", "cdc", datasets.load_cdc_diabetes,
     "data/cdc_diabetes.csv", 1, 0),
    ("Diabetes-130 (readmission)", "d130", datasets.load_diabetes130,
     "data/diabetes_130_hospitals.csv", "Male", "Female"),
    ("BRFSS 2022 (heart disease)", "brfss", datasets.load_brfss,
     "data/brfss2022_subset.csv", 1.0, 2.0),
    ("NHANES 2017-18 (diabetes)", "nhanes", datasets.load_nhanes,
     "data/nhanes_2017_2018.csv", 1, 2),
    ("CCHS 2019-20 (diabetes)", "cchs", datasets.load_cchs,
     "data/cchs_2019_2020_subset.csv", "Male", "Female"),
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep_frames = []
    case_mix_rows = []
    prevalence_rows = []
    threshold_rows = []

    for name, key, loader, relative_path, group_a, group_b in CONFIGS:
        data = loader(path=str(ROOT / relative_path))
        result = run_fairness_analysis(
            df=data["df"],
            feature_cols=data["feature_cols"],
            target_col=data["target_col"],
            group_col=data["group_col"],
            group_a_value=group_a,
            group_b_value=group_b,
            group_labels=data["group_labels"],
            threshold="prevalence",
            random_state=42,
            n_boot=0,
            covariate_blocks=data.get("covariate_blocks"),
            run_calibration_adjustment=False,
            run_case_mix=False,
            verbose=False,
            analysis_label=name,
            cluster_ids=(
                data["df"][data["cluster_col"]] if data.get("cluster_col") else None
            ),
            preprocess_spec=data.get("preprocess_spec"),
        )
        sweep = threshold_sweep(
            result["y_test"], result["prob"], result["g_test"], group_a, group_b
        )
        save_threshold_sweep_plot(
            sweep,
            result["decision_threshold"],
            name,
            FIG_DIR / f"threshold_sweep_{key}.png",
        )
        sweep.insert(0, "dataset", name)
        sweep_frames.append(sweep)

        threshold_rows.append({
            "dataset": name,
            "protected_attribute": data["group_col"],
            "threshold_protocol": result["threshold_protocol"],
            "threshold": result["decision_threshold"],
            "derivation_partition": "training_set",
        })
        for metric in ("ppv", "npv", "predicted_positive_rate"):
            values = result["prevalence_adjustment"][metric]
            prevalence_rows.append({
                "dataset": name,
                "metric": metric,
                "target_prevalence": result["prevalence_adjustment"]["target_prevalence"],
                **values,
            })

        # The design matrix is now built after the split by the
        # training-fitted preprocessor, so the transformed test matrix is
        # taken from the result rather than re-sliced from the loader frame
        # (which still holds raw categorical source columns).
        test_features = result["X_test"]
        for error_type, label in (("fn", "FN_among_positives"),
                                  ("fp", "FP_among_negatives")):
            estimate = covariate_standardized_error_gap(
                result["y_test"], result["pred"], result["g_test"],
                test_features.to_numpy(), group_a, group_b,
                error_type=error_type, seed=42,
            )
            case_mix_rows.append({
                "dataset": name,
                "error_type": label,
                "inference_protocol": "exploratory_point_estimate_no_confidence_interval",
                **estimate,
            })

    pd.concat(sweep_frames, ignore_index=True).to_csv(
        OUT_DIR / "threshold_sweep.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        OUT_DIR / "threshold_policy.csv", index=False
    )
    pd.DataFrame(prevalence_rows).to_csv(
        OUT_DIR / "prevalence_standardized_metrics.csv", index=False
    )
    pd.DataFrame(case_mix_rows).to_csv(
        OUT_DIR / "case_mix_point_estimates.csv", index=False
    )


if __name__ == "__main__":
    main()
