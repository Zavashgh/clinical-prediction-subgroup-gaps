"""
run_age_decomposition.py
========================
Extended fairness decomposition for AGE as the protected attribute.

Mirrors the four-step sex-analysis pipeline:
  1. Raw per-subgroup gaps (youngest vs oldest band)
  2. Prevalence standardization + Shapley decomposition
  3. Case-mix adjustment (g-computation via logistic waterfall; IPW via
     empirical_prevalence_match)
  4. Bootstrap CIs (1,000 resamples)
  5. DerSimonian-Laird random-effects meta-analysis across datasets

Age bands per dataset (from scoping analysis):
  CDC Diabetes  : Age codes 1-5=18-44, 6-9=45-64, 10-13=65+
  Diabetes-130  : age_numeric <30=under_30 (excl. [0-10) n=161),
                  30-59=30_59, >=60=60_plus
  BRFSS 2022    : _AGEG5YR 1-5=18-44, 6-9=45-64, 10-13=65+
  NHANES        : RIDAGEYR <45=under_45, 45-64=45_64, >=65=65_plus
                  (all NHANES age results flagged exploratory)
  CCHS          : DHHGAGE string labels: "18 to 34 years" / "35 to 49 years" /
                  "50 to 64 years" / "65 and older" (12-17 already excluded)

Primary comparison: YOUNGEST vs OLDEST band (gap = youngest - oldest).
All models are blinded to the original age variable (parallel to sex analysis
where sex is excluded from features).

Outputs -> results/extended/age_decomposition/
  age_raw_gaps.csv
  age_bootstrap.csv
  age_shapley.csv
  age_casemix_waterfall.csv
  age_ipw_adjustment.csv
  age_all_bands.csv
  age_meta_analysis_sensitivity_gap.csv
  age_meta_analysis_per_study.csv
"""

import os
import sys
import warnings

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.runtime import (
    configure_deterministic_environment,
    enforce_loaded_threadpool_limits,
)

configure_deterministic_environment()

import numpy as np
import pandas as pd

_RUNTIME_THREAD_LIMITER = enforce_loaded_threadpool_limits()
warnings.filterwarnings("ignore")

from src import datasets as ds
from src import metrics as fm
from src import adjustments as adj
from src.extended_analyses import (
    shapley_decompose_gap,
    empirical_prevalence_match,
    _ci_to_se,
    random_effects_meta,
)
from src.pipeline import run_fairness_analysis
from src.evaluation_splits import (
    isolated_nonprimary_holdout_indices as _nonprimary_test_indices,
    assert_no_training_overlap,
    assert_no_training_cluster_overlap,
)

RESULTS_DIR = os.environ.get(
    "MEDICAL_FAIRNESS_RESULTS_DIR", os.path.join(ROOT, "results")
)
OUT_DIR = os.path.join(RESULTS_DIR, "extended", "age_decomposition")

import re

from src import frozen_outputs as fo

frozen_manifests = []


def _frozen_slug(kind, dataset_name):
    """Filesystem-safe, stable slug for one comparison's frozen artifacts."""
    stem = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    return f"{kind}_{stem}"

os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_BOOT = 1000
AUXILIARY_ONLY = "--auxiliary-only" in sys.argv
JAMA_ONLY = "--jama-only" in sys.argv


# ── DL meta-analysis (calls existing function from extended_analyses) ─────────
def _dl_meta(names, estimates, ses):
    """Thin wrapper returning summary + per-study DataFrames."""
    result = random_effects_meta(names, estimates, ses)
    per_study = result["per_study"]
    summary = {
        "k_datasets":               len(names),
        "pooled_estimate":          result["pooled_estimate"],
        "pooled_se":                result["pooled_se"],
        "pooled_ci_low":            result["pooled_ci_low"],
        "pooled_ci_high":           result["pooled_ci_high"],
        "tau2":                     result["tau2"],
        "Q":                        result["Q"],
        "Q_df":                     result["Q_df"],
        "Q_pvalue":                 result["Q_pvalue"],
        "I2_pct":                   result["I2"],
        "prediction_interval_low":  result["prediction_interval_low"],
        "prediction_interval_high": result["prediction_interval_high"],
    }
    return summary, per_study


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset helpers: add age_band, remove age from features
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_cdc(path="data/cdc_diabetes.csv"):
    """
    CDC Diabetes (BRFSS 2015). Age coded 1-13.
    Bands: 18-44 = codes 1-5, 45-64 = codes 6-9, 65+ = codes 10-13.
    Comparison: '18_44' vs '65_plus'.
    """
    d = ds.load_cdc_diabetes(path)
    df = d["df"]

    age_band_map = {
        1: "18_44", 2: "18_44", 3: "18_44", 4: "18_44", 5: "18_44",
        6: "45_64", 7: "45_64", 8: "45_64", 9: "45_64",
        10: "65_plus", 11: "65_plus", 12: "65_plus", 13: "65_plus",
    }
    df["age_band"] = df["Age"].map(age_band_map)
    df = df[df["age_band"].notna()].copy()

    # Exclude original Age from features (blinded model)
    feature_cols = [c for c in d["feature_cols"] if c != "Age"]

    covariate_blocks = {k: [c for c in v if c != "Age"]
                        for k, v in d["covariate_blocks"].items()}

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": d["target_col"],
        "group_col": "age_band",
        "group_a": "18_44",
        "group_b": "65_plus",
        "all_bands": ["18_44", "45_64", "65_plus"],
        "covariate_blocks": covariate_blocks,
        "preprocess_spec": d["preprocess_spec"],
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "CDC Diabetes (BRFSS 2015)",
        "exploratory": False,
    }


def _prepare_diabetes130(path="data/diabetes_130_hospitals.csv"):
    """
    Diabetes-130. age_numeric midpoints: 5,15,25,35,45,55,65,75,85,95.
    Exclude [0-10) (midpoint 5) entirely.
    Bands: under_30 = midpoints 15,25; 30_59 = 35,45,55; 60_plus = 65+.
    Comparison: 'under_30' vs '60_plus'.
    """
    d = ds.load_diabetes130(path)
    df = d["df"]

    def _band(m):
        if m == 5:
            return np.nan   # exclude [0-10)
        elif m <= 25:
            return "under_30"
        elif m <= 55:
            return "30_59"
        else:
            return "60_plus"

    df["age_band"] = df["age_numeric"].map(_band)
    df = df[df["age_band"].notna()].copy()

    feature_cols = [c for c in d["feature_cols"] if c != "age_numeric"]
    covariate_blocks = {k: [c for c in v if c != "age_numeric"]
                        for k, v in d["covariate_blocks"].items()}

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": d["target_col"],
        "group_col": "age_band",
        "group_a": "under_30",
        "group_b": "60_plus",
        "all_bands": ["under_30", "30_59", "60_plus"],
        "covariate_blocks": covariate_blocks,
        "preprocess_spec": d["preprocess_spec"],
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "Diabetes-130 (readmission)",
        "exploratory": False,
    }


def _prepare_brfss(path="data/brfss2022_subset.csv"):
    """
    BRFSS 2022. _AGEG5YR 1-13 (14=refused/missing excluded).
    Bands: 18-44 = 1-5, 45-64 = 6-9, 65+ = 10-13.
    Comparison: '18_44' vs '65_plus'.
    """
    d = ds.load_brfss(path)
    df = d["df"]

    # _AGEG5YR is still in the dataframe (not dropped by loader)
    age_band_map = {
        1: "18_44", 2: "18_44", 3: "18_44", 4: "18_44", 5: "18_44",
        6: "45_64", 7: "45_64", 8: "45_64", 9: "45_64",
        10: "65_plus", 11: "65_plus", 12: "65_plus", 13: "65_plus",
        14: np.nan,
    }
    df["age_band"] = df["_AGEG5YR"].map(age_band_map)
    df = df[df["age_band"].notna()].copy()

    # Exclude age_numeric from features
    feature_cols = [c for c in d["feature_cols"] if c != "age_numeric"]
    covariate_blocks = {k: [c for c in v if c != "age_numeric"]
                        for k, v in d["covariate_blocks"].items()}

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": d["target_col"],
        "group_col": "age_band",
        "group_a": "18_44",
        "group_b": "65_plus",
        "all_bands": ["18_44", "45_64", "65_plus"],
        "covariate_blocks": covariate_blocks,
        "preprocess_spec": d["preprocess_spec"],
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "BRFSS 2022 (heart disease)",
        "exploratory": False,
    }


def _prepare_nhanes(path="data/nhanes_2017_2018.csv"):
    """
    NHANES 2017-18. RIDAGEYR continuous (adults >=18 already filtered).
    Bands: <45 = 'under_45', 45-64 = '45_64', >=65 = '65_plus'.
    Comparison: 'under_45' vs '65_plus'.
    NOTE: under_45 has 104 positives in the full prepared dataset, but only
    26-36 test-set positives across seeds 0-9 -> all results exploratory.
    """
    d = ds.load_nhanes(path)
    df = d["df"]

    # age_years is RIDAGEYR; create bands
    df["age_band"] = pd.cut(
        df["age_years"],
        bins=[0, 45, 65, 200],
        labels=["under_45", "45_64", "65_plus"],
        right=False,
    ).astype(str)
    df = df[df["age_band"].isin(["under_45", "45_64", "65_plus"])].copy()

    feature_cols = [c for c in d["feature_cols"] if c != "age_years"]
    covariate_blocks = {k: [c for c in v if c != "age_years"]
                        for k, v in d["covariate_blocks"].items()}

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": d["target_col"],
        "group_col": "age_band",
        "group_a": "under_45",
        "group_b": "65_plus",
        "all_bands": ["under_45", "45_64", "65_plus"],
        "covariate_blocks": covariate_blocks,
        "preprocess_spec": d["preprocess_spec"],
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "NHANES 2017-18 (diabetes)",
        "exploratory": True,  # under_45 has only 26-36 test positives per split
    }


def _prepare_cchs(path="data/cchs_2019_2020_subset.csv"):
    """
    CCHS 2019-20. DHHGAGE string: 5 levels, 12-17 already excluded by loader.
    Bands: '18 to 34 years', '35 to 49 years', '50 to 64 years', '65 and older'.
    Comparison: '18 to 34 years' vs '65 and older'.
    """
    d = ds.load_cchs(path)
    df = d["df"]

    # DHHGAGE is still in the dataframe; age_years was derived from it by loader
    # Re-map string labels to tidy band names
    cchs_band_map = {
        "18 to 34 years":  "18_34",
        "35 to 49 years":  "35_49",
        "50 to 64 years":  "50_64",
        "65 and older":    "65_plus",
    }
    df["age_band"] = df["DHHGAGE"].map(cchs_band_map)
    df = df[df["age_band"].notna()].copy()

    feature_cols = [c for c in d["feature_cols"] if c != "age_years"]
    covariate_blocks = {k: [c for c in v if c != "age_years"]
                        for k, v in d["covariate_blocks"].items()}

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": d["target_col"],
        "group_col": "age_band",
        "group_a": "18_34",
        "group_b": "65_plus",
        "all_bands": ["18_34", "35_49", "50_64", "65_plus"],
        "covariate_blocks": covariate_blocks,
        "preprocess_spec": d["preprocess_spec"],
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "CCHS 2019-20 (diabetes)",
        "exploratory": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All-bands metrics (no model refitting; computed from test-set arrays
# returned by the primary pipeline run)
# ─────────────────────────────────────────────────────────────────────────────

def _all_bands_metrics(y_test, prob, pred, g_test, all_bands, dataset_name,
                       primary_groups, best_model, decision_threshold):
    """
    Compute threshold-derived metrics for every age band on the test set.
    Uses the model already fitted on youngest vs oldest data, applied to all
    rows in the test set that belong to any band (including the middle band).
    """
    rows = []
    for band in all_bands:
        mask = g_test == band
        if mask.sum() == 0:
            continue
        m = fm.threshold_metrics(y_test[mask], pred[mask])
        rows.append({
            "dataset": dataset_name,
            "age_band": band,
            "evaluation_partition": (
                "primary_heldout_test" if band in primary_groups
                else "independent_nonprimary_holdout"
            ),
            "model_training_groups": "|".join(map(str, primary_groups)),
            "best_model": best_model,
            "decision_threshold": decision_threshold,
            "training_overlap_n": 0,
            "n": m["n"],
            "n_pos": m["n_pos"],
            "prevalence": m["prevalence"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "fnr": m["fnr"],
            "fpr": m["fpr"],
            "ppv": m["ppv"],
            "npv": m["npv"],
            "predicted_positive_rate": m["predicted_positive_rate"],
        })
    return pd.DataFrame(rows)


def _atomic_write_csv(frame, path):
    temp_path = f"{path}.tmp"
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = [
    ("data/cdc_diabetes.csv",              _prepare_cdc),
    ("data/diabetes_130_hospitals.csv",    _prepare_diabetes130),
    ("data/brfss2022_subset.csv",          _prepare_brfss),
    ("data/nhanes_2017_2018.csv",          _prepare_nhanes),
    ("data/cchs_2019_2020_subset.csv",     _prepare_cchs),
]

raw_gap_rows        = []
boot_rows           = []
shapley_rows        = []
casemix_rows        = []
ipw_rows            = []
all_bands_rows      = []
meta_inputs         = []   # (name, sens_est, sens_se, exploratory)
completed_datasets  = []
run_errors          = []

for path, loader_fn in LOADERS:
    print(f"\n{'='*70}")
    print(f"  {path}")
    print(f"{'='*70}")

    try:
        dset = loader_fn(path)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        run_errors.append(f"{path}: load error: {e}")
        continue

    name     = dset["name"]
    df       = dset["df"]
    fc       = dset["feature_cols"]
    tc       = dset["target_col"]
    gc       = dset["group_col"]
    ga       = dset["group_a"]
    gb       = dset["group_b"]
    cov_blk  = dset["covariate_blocks"]
    is_expl  = dset["exploratory"]
    all_bnds = dset["all_bands"]

    # ── print subgroup sizes ────────────────────────────────────────────
    for band in all_bnds:
        sub = df[df[gc] == band]
        n = len(sub); n_pos = int(sub[tc].sum())
        print(f"  Band '{band}': n={n:,}, positives={n_pos:,} "
              f"(prev={n_pos/n:.3f})")

    # ── run primary pipeline (youngest vs oldest) ──────────────────────
    threshold = "prevalence"   # same convention as sex analysis for low-prev outcomes
    try:
        result = run_fairness_analysis(
            df=df,
            feature_cols=fc,
            target_col=tc,
            group_col=gc,
            group_a_value=ga,
            group_b_value=gb,
            group_labels={ga: f"Youngest ({ga})", gb: f"Oldest ({gb})"},
            threshold=threshold,
            test_size=0.3,
            random_state=SEED,
            n_boot=N_BOOT,
            covariate_blocks=cov_blk,
            run_calibration_adjustment=False,
            run_case_mix=True,
            verbose=True,
            analysis_label=name,
            cluster_ids=dset.get("cluster_ids"),
            preprocess_spec=dset.get("preprocess_spec"),
        )
    except Exception as e:
        print(f"  PIPELINE ERROR: {e}")
        import traceback; traceback.print_exc()
        run_errors.append(f"{path}: pipeline error: {e}")
        continue

    # ── unpack test-set arrays ─────────────────────────────────────────
    y_te   = result["y_test"]
    prob_c = result["prob"]
    pred_c = result["pred"]
    g_te   = result["g_test"]

    metrics_a = result["subgroup_metrics"]["A"]   # youngest
    metrics_b = result["subgroup_metrics"]["B"]   # oldest

    # ── raw gap row ────────────────────────────────────────────────────
    boot_sens  = result["bootstrap_raw"]["sensitivity"]
    boot_ppv   = result["bootstrap_raw"]["ppv"]
    boot_fnr   = result["bootstrap_raw"]["fnr"]
    boot_ppr   = result["bootstrap_raw"]["predicted_positive_rate"]
    boot_ppv_a = result["bootstrap_prevalence_adjusted"]["ppv"]

    raw_gap_rows.append({
        "dataset":              name,
        "group_a":              ga,
        "group_b":              gb,
        "n_a":                  metrics_a["n"],
        "n_b":                  metrics_b["n"],
        "n_pos_a":              metrics_a["n_pos"],
        "n_pos_b":              metrics_b["n_pos"],
        "prev_a":               metrics_a["prevalence"],
        "prev_b":               metrics_b["prevalence"],
        "auroc":                result["test_auroc"][result["best_model_name"]],
        "best_model":           result["best_model_name"],
        "sensitivity_a":        metrics_a["sensitivity"],
        "sensitivity_b":        metrics_b["sensitivity"],
        "sensitivity_gap":      boot_sens["point"],
        "sensitivity_ci_low":   boot_sens["ci_low"],
        "sensitivity_ci_high":  boot_sens["ci_high"],
        "fnr_a":                metrics_a["fnr"],
        "fnr_b":                metrics_b["fnr"],
        "fnr_gap":              boot_fnr["point"],
        "fnr_ci_low":           boot_fnr["ci_low"],
        "fnr_ci_high":          boot_fnr["ci_high"],
        "ppv_raw_a":            metrics_a["ppv"],
        "ppv_raw_b":            metrics_b["ppv"],
        "ppv_gap_raw":          boot_ppv["point"],
        "ppv_gap_raw_ci_low":   boot_ppv["ci_low"],
        "ppv_gap_raw_ci_high":  boot_ppv["ci_high"],
        "ppr_a":                metrics_a["predicted_positive_rate"],
        "ppr_b":                metrics_b["predicted_positive_rate"],
        "ppr_gap":              boot_ppr["point"],
        "ppr_ci_low":           boot_ppr["ci_low"],
        "ppr_ci_high":          boot_ppr["ci_high"],
        "ppv_gap_adj":          boot_ppv_a["point"],
        "ppv_gap_adj_ci_low":   boot_ppv_a["ci_low"],
        "ppv_gap_adj_ci_high":  boot_ppv_a["ci_high"],
        "ppv_attenuation_pct":  result["prevalence_adjustment"]["ppv"]["attenuation_pct"],
        "exploratory":          is_expl,
        "note": ("NHANES age: sensitivity gap was negative in 10/10 seeds under both "
                 "legacy fixed-logistic and corrected model-selection protocols; "
                 "magnitude remains highly imprecise (26-36 under_45 test positives "
                 "per split)"
                 if is_expl else ""),
    })

    # ── bootstrap rows (compact) ───────────────────────────────────────
    for metric_key, boot_res in result["bootstrap_raw"].items():
        boot_rows.append({
            "dataset": name,
            "comparison": f"{ga}_vs_{gb}",
            "metric": metric_key,
            "estimate": boot_res["point"],
            "ci_low": boot_res["ci_low"],
            "ci_high": boot_res["ci_high"],
            "type": "raw",
        })
    for metric_key, boot_res in result["bootstrap_prevalence_adjusted"].items():
        boot_rows.append({
            "dataset": name,
            "comparison": f"{ga}_vs_{gb}",
            "metric": metric_key,
            "estimate": boot_res["point"],
            "ci_low": boot_res["ci_low"],
            "ci_high": boot_res["ci_high"],
            "type": "prevalence_adjusted",
        })

    # ── Shapley decomposition ──────────────────────────────────────────
    shap_df = shapley_decompose_gap(metrics_a, metrics_b)
    shap_df.insert(0, "dataset", name)
    shap_df.insert(1, "comparison", f"{ga}_vs_{gb}")
    shapley_rows.append(shap_df)

    # ── case-mix waterfall ────────────────────────────────────────────
    if "case_mix" in result and result["case_mix"] is not None:
        cm = result["case_mix"].copy()
        cm.insert(0, "dataset", name)
        cm.insert(1, "comparison", f"{ga}_vs_{gb}")
        casemix_rows.append(cm)
    else:
        raise RuntimeError(f"{name}: required case-mix result was not produced")

    # ── IPW prevalence matching ────────────────────────────────────────
    mask_a_te = g_te == ga
    mask_b_te = g_te == gb
    target_prev = result["prevalence_adjustment"]["target_prevalence"]

    try:
        ipw_result = empirical_prevalence_match(
            y_te, prob_c, pred_c, g_te,
            ga, gb,
            target_prevalence=target_prev,
            n_boot=0 if JAMA_ONLY else 500, seed=SEED,
        )
        ipw_tbl = ipw_result["table"].copy()
        ipw_tbl.insert(0, "dataset", name)
        ipw_tbl.insert(1, "comparison", f"{ga}_vs_{gb}")
        ipw_rows.append(ipw_tbl)
    except Exception as e:
        raise RuntimeError(f"{name}: required IPW result failed: {e}") from e

    # ── all-bands metrics (on the test set rows belonging to any band) ─
    # Preserve the primary model and its exact youngest/oldest held-out rows.
    # Each non-primary band gets a separate reproducible 30% holdout. The
    # combined auxiliary evaluation is therefore disjoint from model training.
    try:
        work_all = df[df[gc].isin(all_bnds)].copy()
        primary_train_index = pd.Index(result["train_index"])
        primary_test_index = pd.Index(result["test_index"])
        nonprimary_bands = [band for band in all_bnds if band not in (ga, gb)]

        # With a clustering identifier, row-level disjointness is not enough:
        # a non-primary-band encounter can belong to a patient who trained the
        # model. Restrict the non-primary pool to patients unseen in training.
        cluster_ids = dset.get("cluster_ids")
        if cluster_ids is not None:
            training_clusters = set(cluster_ids.reindex(primary_train_index))
            pool_clusters = cluster_ids.reindex(work_all.index)
            nonprimary_pool = work_all[~pool_clusters.isin(training_clusters)]
        else:
            nonprimary_pool = work_all

        nonprimary_test_index = _nonprimary_test_indices(
            nonprimary_pool, tc, gc, nonprimary_bands,
            test_size=0.3, random_state=SEED,
        )
        evaluation_index = primary_test_index.append(nonprimary_test_index)
        if evaluation_index.has_duplicates:
            raise RuntimeError(f"{name}: duplicate rows in all-bands evaluation set")
        assert_no_training_overlap(primary_train_index, evaluation_index, context=name)
        if cluster_ids is not None:
            assert_no_training_cluster_overlap(
                cluster_ids.reindex(primary_train_index),
                cluster_ids.reindex(evaluation_index),
                context=f"{name} all-bands",
            )

        evaluation = work_all.loc[evaluation_index]
        # The calibrated model is a pipeline whose first step is the
        # training-fitted preprocessing, so it is given RAW rows here. Passing a
        # pre-transformed matrix would bypass the fold-specific preprocessors
        # the calibrated ensemble carries.
        y_te_all = evaluation[tc].astype(int)
        g_te_all = evaluation[gc]
        calib_model = result["calibrated_model"]
        thr_all = float(result["decision_threshold"])
        prob_all = calib_model.predict_proba(evaluation)[:, 1]
        pred_all = (prob_all >= thr_all).astype(int)

        ab_df = _all_bands_metrics(
            y_te_all.values, prob_all, pred_all,
            g_te_all.values, all_bnds, name,
            primary_groups=(ga, gb),
            best_model=result["best_model_name"],
            decision_threshold=thr_all,
        )
        if len(ab_df) != len(all_bnds) or int(ab_df["training_overlap_n"].sum()) != 0:
            raise RuntimeError(f"{name}: incomplete or contaminated all-bands output")
        all_bands_rows.append(ab_df)
    except Exception as e:
        raise RuntimeError(f"{name}: required all-bands metrics failed: {e}") from e

    # ── frozen held-out outputs ────────────────────────────────────────
    # Persist this comparison's row-level held-out predictions, its
    # discrimination/calibration table, and its fitted objects. Read-only with
    # respect to the analysis: every value comes from `result` as computed above.
    try:
        frozen_manifests.append(
            fo.write_comparison_artifacts(
                result,
                slug=_frozen_slug("age", name),
                dataset=name,
                comparison_type="age",
                comparison=f"{ga}_vs_{gb}",
                results_dir=RESULTS_DIR,
                cluster_ids=dset.get("cluster_ids"),
            )
        )
    except Exception as e:
        raise RuntimeError(f"{name}: frozen output export failed: {e}") from e

    # ── collect meta-analysis input ───────────────────────────────────
    sens_est = boot_sens["point"]
    # Derive SE consistently across sex, age, and race from the bootstrap CI.
    sens_se = float(_ci_to_se(boot_sens["ci_low"], boot_sens["ci_high"]))
    meta_inputs.append((name, sens_est, sens_se, is_expl))
    completed_datasets.append(name)

    print(f"\n  Sensitivity gap ({ga} - {gb}): {sens_est:+.4f} "
          f"(95% CI [{boot_sens['ci_low']:+.4f}, {boot_sens['ci_high']:+.4f}])")
    print(f"  PPV gap raw={boot_ppv['point']:+.4f}, adj={boot_ppv_a['point']:+.4f}, "
          f"attenuation={result['prevalence_adjustment']['ppv']['attenuation_pct']:.1f}%")
    if is_expl:
        print("  *** EXPLORATORY: small subgroup sizes, interpret with caution ***")


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary-only production mode replaces only age_all_bands.csv after all
# five datasets succeed. No primary or meta-analysis file is touched.
if AUXILIARY_ONLY:
    expected_datasets = len(LOADERS)
    if (run_errors or len(all_bands_rows) != expected_datasets
            or len(completed_datasets) != expected_datasets):
        raise RuntimeError(
            "Age auxiliary rerun incomplete; no result files were written. "
            f"Completed {len(completed_datasets)}/{expected_datasets}; "
            f"all-bands tables={len(all_bands_rows)}; errors={run_errors}"
        )
    auxiliary = pd.concat(all_bands_rows, ignore_index=True)
    auxiliary_path = os.path.join(OUT_DIR, "age_all_bands.csv")
    _atomic_write_csv(auxiliary, auxiliary_path)
    print(f"\n[OK] Saved only {auxiliary_path}")
    raise SystemExit(0)


# Refuse to write a partial production result set. Every required component
# for every configured dataset must exist before the first CSV is replaced.
expected_datasets = len(LOADERS)
component_counts = {
    "raw gaps": len(raw_gap_rows),
    "Shapley": len(shapley_rows),
    "case mix": len(casemix_rows),
    "IPW": len(ipw_rows),
    "all bands": len(all_bands_rows),
    "meta inputs": len(meta_inputs),
}
incomplete = {k: v for k, v in component_counts.items() if v != expected_datasets}
if run_errors or len(completed_datasets) != expected_datasets or incomplete:
    details = "; ".join(run_errors) if run_errors else "no explicit dataset error recorded"
    raise RuntimeError(
        "Age rerun incomplete; no result files were written. "
        f"Completed {len(completed_datasets)}/{expected_datasets}; "
        f"component counts={component_counts}; errors={details}"
    )


# Save per-dataset tables
# ─────────────────────────────────────────────────────────────────────────────

pd.DataFrame(raw_gap_rows).to_csv(
    os.path.join(OUT_DIR, "age_raw_gaps.csv"), index=False)
print("\nSaved age_raw_gaps.csv")

pd.DataFrame(boot_rows).to_csv(
    os.path.join(OUT_DIR, "age_bootstrap.csv"), index=False)
print("Saved age_bootstrap.csv")

if shapley_rows and not JAMA_ONLY:
    pd.concat(shapley_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "age_shapley.csv"), index=False)
    print("Saved age_shapley.csv")

if casemix_rows:
    pd.concat(casemix_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "age_casemix_waterfall.csv"), index=False)
    print("Saved age_casemix_waterfall.csv")

if ipw_rows:
    pd.concat(ipw_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "age_ipw_adjustment.csv"), index=False)
    print("Saved age_ipw_adjustment.csv")

if all_bands_rows and not JAMA_ONLY:
    pd.concat(all_bands_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "age_all_bands.csv"), index=False)
    print("Saved age_all_bands.csv")

if JAMA_ONLY:
    print("\n[OK] Saved JAMA-scoped age outputs only:", OUT_DIR)
    raise SystemExit(0)


# ─────────────────────────────────────────────────────────────────────────────
# DerSimonian-Laird meta-analysis of sensitivity gap (youngest vs oldest)
# Run once including all 5 datasets, and once excluding NHANES (exploratory)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("DerSimonian-Laird meta-analysis: sensitivity gap (youngest vs oldest)")
print("="*70)

names_all = [t[0] for t in meta_inputs]
ests_all  = np.array([t[1] for t in meta_inputs])
ses_all   = np.array([t[2] for t in meta_inputs])

summary_all, per_study_all = _dl_meta(names_all, ests_all, ses_all)
print(f"All 5 datasets:")
print(f"  Pooled = {summary_all['pooled_estimate']:+.4f} "
      f"(95% CI [{summary_all['pooled_ci_low']:+.4f}, "
      f"{summary_all['pooled_ci_high']:+.4f}])")
print(f"  PI = [{summary_all['prediction_interval_low']:+.4f}, "
      f"{summary_all['prediction_interval_high']:+.4f}]")
print(f"  I^2 = {summary_all['I2_pct']:.1f}%  tau^2 = {summary_all['tau2']:.5f}")

# Excluding NHANES
non_expl = [(t[0], t[1], t[2]) for t in meta_inputs if not t[3]]
if len(non_expl) >= 2:
    names_ne = [t[0] for t in non_expl]
    ests_ne  = np.array([t[1] for t in non_expl])
    ses_ne   = np.array([t[2] for t in non_expl])
    summary_ne, per_study_ne = _dl_meta(names_ne, ests_ne, ses_ne)
    print(f"\nExcluding NHANES (sensitivity analysis):")
    print(f"  Pooled = {summary_ne['pooled_estimate']:+.4f} "
          f"(95% CI [{summary_ne['pooled_ci_low']:+.4f}, "
          f"{summary_ne['pooled_ci_high']:+.4f}])")

# Build combined summary CSV
meta_summary_rows = []
for label, summ, per_st in [
    ("all_5_datasets",          summary_all, per_study_all),
    ("excl_nhanes_exploratory", summary_ne if len(non_expl) >= 2 else None, None),
]:
    if summ is None:
        continue
    row = {"analysis": label, **summ}
    meta_summary_rows.append(row)

pd.DataFrame(meta_summary_rows).to_csv(
    os.path.join(OUT_DIR, "age_meta_analysis_sensitivity_gap.csv"), index=False)
print("\nSaved age_meta_analysis_sensitivity_gap.csv")

per_study_all["analysis"] = "all_5_datasets"
if len(non_expl) >= 2:
    per_study_ne["analysis"] = "excl_nhanes_exploratory"
    per_study_combined = pd.concat([per_study_all, per_study_ne], ignore_index=True)
else:
    per_study_combined = per_study_all

per_study_combined.to_csv(
    os.path.join(OUT_DIR, "age_meta_analysis_per_study.csv"), index=False)
print("Saved age_meta_analysis_per_study.csv")

print("\nAll age decomposition outputs saved to:", OUT_DIR)
