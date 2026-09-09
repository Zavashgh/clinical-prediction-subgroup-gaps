"""
run_race_decomposition.py
=========================
Extended fairness decomposition for RACE/ETHNICITY as the protected attribute.

Second protected-attribute extension, following the identical four-step
framework used for sex and age:
  1. Raw per-subgroup gaps (minority vs reference-majority group)
  2. Prevalence standardization + Shapley decomposition
  3. Case-mix adjustment (g-computation via logistic waterfall; IPW)
  4. Bootstrap CIs (1,000 resamples)
  5. DerSimonian-Laird random-effects meta-analysis across datasets

DATASETS WITH USABLE RACE DATA (3 of 5):
  Diabetes-130  : reconstructed race group; AfricanAmerican vs Caucasian
  BRFSS 2022    : `race` (from _RACEGR4); Black vs White (primary),
                  Hispanic vs White and Multiracial vs White (secondary)
  NHANES        : `race` (from RIDRETH3); Black vs White  --  EXPLORATORY
                  (marginal subgroup sizes, consistent with sex/age treatment)

DATASETS EXPLICITLY EXCLUDED (no usable race column):
  CDC Diabetes (BRFSS 2015)  -- no race/ethnicity variable in the file
  CCHS 2019-20               -- PUMF subset carries no race/ethnicity variable

Gap convention: A - B, with A = minority group, B = reference-majority group,
so the sign is consistent across datasets (minority minus majority-reference)
for a coherent pooled meta-analytic estimate. This parallels the sex analysis
(Male - Female) and age analysis (youngest - oldest).

Models are BLINDED to the race variable: all race one-hot dummies are removed
from the feature set and from the case-mix covariate blocks (parallel to sex
and age, where the protected attribute is excluded from features).

Per-study SE for the meta-analysis is derived with `_ci_to_se` (CI width /
2*1.95996) -- the SAME method used in the sex and age analyses -- so all three
protected-attribute meta-analyses are methodologically consistent from the
start.

Small-subgroup rule: for any group with < ~100 test-set positive cases, a
10-seed stability check on the sensitivity-gap DIRECTION is run automatically
(the same legacy fixed-logistic check used for the original race evidence).
The prespecified Hispanic-vs-White follow-up is also recomputed even though it
is above that threshold, so its saved 10/10-negative evidence remains
reproducible rather than depending on a manually appended CSV row.

Outputs -> results/extended/race_decomposition/
  race_raw_gaps.csv
  race_bootstrap.csv
  race_shapley.csv
  race_casemix_waterfall.csv
  race_ipw_adjustment.csv
  race_all_groups.csv
  race_secondary_comparisons.csv
  race_stability_checks.csv
  race_meta_analysis_sensitivity_gap.csv
  race_meta_analysis_per_study.csv
  NOTES.md  (written separately)
"""

import os
import sys
import warnings

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
    random_effects_meta,
    _ci_to_se,
)
from src.pipeline import run_fairness_analysis

from src.evaluation_splits import (
    isolated_nonprimary_holdout_indices as _nonprimary_test_indices,
    assert_no_training_overlap,
    assert_no_training_cluster_overlap,
)

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

RESULTS_DIR = os.environ.get(
    "MEDICAL_FAIRNESS_RESULTS_DIR", os.path.join(ROOT, "results")
)
OUT_DIR = os.path.join(RESULTS_DIR, "extended", "race_decomposition")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_BOOT = 1000
AUXILIARY_ONLY = "--auxiliary-only" in sys.argv
JAMA_ONLY = "--jama-only" in sys.argv
STABILITY_POS_THRESHOLD = 100   # run 10-seed check if a group has < this many test positives
FORCED_STABILITY_COMPARISONS = {
    # Prespecified follow-up already persisted in commit 122272e. Recompute it
    # during every production run so the saved evidence is not silently lost.
    ("BRFSS 2022 (heart disease)", "Hispanic", "White"),
}
EXPECTED_SECONDARY_COMPARISONS = 2


# ── DL meta-analysis wrapper (identical to age analysis) ──────────────────────
def _dl_meta(names, estimates, ses):
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


# ── 10-seed stability check (mirrors the age Diabetes-130 check) ──────────────
def stability_check_10seed(df, fc, tc, gc, ga, gb, preprocess_spec=None,
                           cluster_ids=None):
    """
    Legacy fixed-logistic protocol: refit a logistic-regression + isotonic
    pipeline on 10 different 70/30 splits (seeds 0-9), use the training
    prevalence threshold, and record the sensitivity gap (A - B) each time.
    This is a non-leaky stability check, but it does not repeat model-family
    selection at each seed. Returns a dict summarising direction stability.

    Corrected path: the design matrix is built INSIDE the seed loop by a
    preprocessor fitted on that seed's training rows only, so each replicate
    is independently free of preprocessing leakage. Where a clustering
    identifier exists, the per-seed split and the isotonic calibration folds
    are patient-grouped, matching the primary analysis.
    """
    from src.pipeline import _make_calibrated_classifier
    from src.preprocessing import TrainFittedPreprocessor

    work = df[df[gc].isin([ga, gb])].copy()
    y = work[tc].astype(int)
    g = work[gc]
    spec = (preprocess_spec or {}).get("categorical", [])
    clusters = None
    if cluster_ids is not None:
        clusters = cluster_ids.reindex(work.index)

    gaps = []
    for seed in range(10):
        if clusters is None:
            tr_idx, te_idx = train_test_split(
                work.index, test_size=0.3, random_state=seed, stratify=y
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.3, random_state=seed
            )
            pos_tr, pos_te = next(
                splitter.split(work.index, y, groups=clusters.to_numpy())
            )
            tr_idx, te_idx = work.index[pos_tr], work.index[pos_te]

        pre = TrainFittedPreprocessor(fc, spec).fit(work.loc[tr_idx])
        X_tr = pre.transform(work.loc[tr_idx], partition="stability_train")
        X_te = pre.transform(work.loc[te_idx], partition="stability_test")
        y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
        g_te = g.loc[te_idx]

        thr = float(y_tr.mean())
        model = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(
                              max_iter=2000, C=1.0, random_state=seed,
                          ))])
        if clusters is None:
            calibration_cv = 5
        else:
            calibration_cv = list(
                StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                .split(X_tr, y_tr, groups=clusters.loc[tr_idx].to_numpy())
            )
        cal = _make_calibrated_classifier(model, cv=calibration_cv)
        cal.fit(X_tr, y_tr)
        prob = cal.predict_proba(X_te)[:, 1]
        pred = (prob >= thr).astype(int)
        mask_a = (g_te == ga).values
        mask_b = (g_te == gb).values
        ma = fm.threshold_metrics(y_te.values[mask_a], pred[mask_a])
        mb = fm.threshold_metrics(y_te.values[mask_b], pred[mask_b])
        gaps.append(ma["sensitivity"] - mb["sensitivity"])

    gaps = np.array(gaps)
    n_pos_dir = int((gaps > 0).sum())
    n_neg_dir = int((gaps < 0).sum())
    majority_dir = "positive" if n_pos_dir > n_neg_dir else "negative"
    n_majority = max(n_pos_dir, n_neg_dir)
    return {
        "protocol":       "legacy_fixed_logistic_isotonic",
        "mean_gap":       float(gaps.mean()),
        "sd_gap":         float(gaps.std()),
        "min_gap":        float(gaps.min()),
        "max_gap":        float(gaps.max()),
        "n_positive_dir": n_pos_dir,
        "n_negative_dir": n_neg_dir,
        "majority_direction": majority_dir,
        "n_seeds_majority": n_majority,
        "stable": bool(n_majority >= 9),   # >=9/10 same sign = directionally stable
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset preparation: build race_group column, strip race from features
# ─────────────────────────────────────────────────────────────────────────────

def _is_race_feature(column):
    """True for the raw categorical source AND any pre-expanded indicator.

    Under the corrected path the loaders expose ``race`` as a raw categorical
    source column rather than pre-built ``race_*`` dummies, so the blinding
    rule has to exclude the source name too. Both spellings are matched so
    the intended exclusion holds regardless of which form is present.
    """
    lowered = column.lower()
    return lowered == "race" or lowered.startswith("race_")


def _strip_race(feature_cols, covariate_blocks, preprocess_spec=None):
    """Remove race from the predictors, the covariate blocks, and the
    preprocessing spec, so the model is blinded to race and the
    training-fitted preprocessor never learns a race schema for this
    comparison."""
    fc = [c for c in feature_cols if not _is_race_feature(c)]
    cov = {k: [c for c in v if not _is_race_feature(c)]
           for k, v in covariate_blocks.items()}
    spec = dict(preprocess_spec or {})
    spec["categorical"] = [
        entry for entry in spec.get("categorical", [])
        if not _is_race_feature(entry["source"])
    ]
    return fc, cov, spec


def _prepare_diabetes130(path="data/diabetes_130_hospitals.csv"):
    """
    Diabetes-130. The loader now retains the raw `race` column, so the group
    label is read directly instead of being reconstructed from one-hot
    dummies. The resulting groups are identical to the previous
    reconstruction (AfricanAmerican was the dropped reference level).
    Primary: AfricanAmerican (A) vs Caucasian (B).
    """
    d = ds.load_diabetes130(path)
    df = d["df"]

    df["race_group"] = df["race"]

    fc, cov, spec = _strip_race(
        d["feature_cols"], d["covariate_blocks"], d["preprocess_spec"]
    )

    return {
        "df": df,
        "feature_cols": fc,
        "target_col": d["target_col"],
        "group_col": "race_group",
        "group_a": "AfricanAmerican",
        "group_b": "Caucasian",
        "all_groups": ["AfricanAmerican", "Caucasian", "Hispanic", "Asian", "Other", "Unknown"],
        "covariate_blocks": cov,
        "preprocess_spec": spec,
        "cluster_col": d.get("cluster_col"),
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "Diabetes-130 (readmission)",
        "exploratory": False,
        "secondary": [],
    }


def _prepare_brfss(path="data/brfss2022_subset.csv"):
    """
    BRFSS 2022. `race` column mapped from _RACEGR4 by the loader:
      1=White, 2=Black, 3=OtherRace, 4=Multiracial, 5=Hispanic, 9=Unknown.
    Primary: Black (A) vs White (B).
    Secondary: Hispanic vs White, Multiracial vs White.
    """
    d = ds.load_brfss(path)
    df = d["df"]

    fc, cov, spec = _strip_race(
        d["feature_cols"], d["covariate_blocks"], d["preprocess_spec"]
    )

    return {
        "df": df,
        "feature_cols": fc,
        "target_col": d["target_col"],
        "group_col": "race",
        "group_a": "Black",
        "group_b": "White",
        "all_groups": ["White", "Black", "Hispanic", "Multiracial", "OtherRace", "Unknown"],
        "covariate_blocks": cov,
        "preprocess_spec": spec,
        "cluster_col": d.get("cluster_col"),
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "BRFSS 2022 (heart disease)",
        "exploratory": False,
        "secondary": [("Hispanic", "White"), ("Multiracial", "White")],
    }


def _prepare_nhanes(path="data/nhanes_2017_2018.csv"):
    """
    NHANES 2017-18. `race` from RIDRETH3 (adults >=18 filtered by loader).
    Primary: Black (A) vs White (B). EXPLORATORY throughout (marginal sizes).
    """
    d = ds.load_nhanes(path)
    df = d["df"]

    fc, cov, spec = _strip_race(
        d["feature_cols"], d["covariate_blocks"], d["preprocess_spec"]
    )

    return {
        "df": df,
        "feature_cols": fc,
        "target_col": d["target_col"],
        "group_col": "race",
        "group_a": "Black",
        "group_b": "White",
        "all_groups": ["White", "Black", "MexicanAmerican", "OtherHispanic",
                       "Asian", "OtherMultiracial"],
        "covariate_blocks": cov,
        "preprocess_spec": spec,
        "cluster_col": d.get("cluster_col"),
        "cluster_ids": (
            df[d["cluster_col"]] if d.get("cluster_col") else None
        ),
        "name": "NHANES 2017-18 (diabetes)",
        "exploratory": True,
        "secondary": [],
    }


# ── all-groups metrics ────────────────────────────────────────────────────────
def _all_groups_metrics(y_test, prob, pred, g_test, all_groups, dataset_name,
                        primary_groups, best_model, decision_threshold):
    rows = []
    for grp in all_groups:
        mask = g_test == grp
        if mask.sum() == 0:
            continue
        m = fm.threshold_metrics(y_test[mask], pred[mask])
        rows.append({
            "dataset": dataset_name, "race_group": grp,
            "evaluation_partition": (
                "primary_heldout_test" if grp in primary_groups
                else "independent_nonprimary_holdout"
            ),
            "model_training_groups": "|".join(map(str, primary_groups)),
            "best_model": best_model,
            "decision_threshold": decision_threshold,
            "training_overlap_n": 0,
            "n": m["n"], "n_pos": m["n_pos"], "prevalence": m["prevalence"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "fnr": m["fnr"], "fpr": m["fpr"], "ppv": m["ppv"], "npv": m["npv"],
            "predicted_positive_rate": m["predicted_positive_rate"],
        })
    return pd.DataFrame(rows)


def _atomic_write_csv(frame, path):
    temp_path = f"{path}.tmp"
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


# ── one full comparison (used for both primary and secondary) ─────────────────
def run_one_comparison(dset, ga, gb, is_secondary=False):
    """Run the full pipeline for one A-vs-B race comparison; return a dict of
    result artefacts (raw-gap row, bootstrap rows, shapley df, casemix df,
    ipw df, meta input tuple)."""
    name = dset["name"]
    df   = dset["df"]
    fc   = dset["feature_cols"]
    tc   = dset["target_col"]
    gc   = dset["group_col"]
    cov  = dset["covariate_blocks"]
    is_expl = dset["exploratory"]

    result = run_fairness_analysis(
        df=df, feature_cols=fc, target_col=tc, group_col=gc,
        group_a_value=ga, group_b_value=gb,
        group_labels={ga: ga, gb: gb},
        threshold="prevalence", test_size=0.3, random_state=SEED,
        n_boot=N_BOOT, covariate_blocks=cov,
        run_calibration_adjustment=False, run_case_mix=True, verbose=True,
        analysis_label=name,
        cluster_ids=dset.get("cluster_ids"),
        preprocess_spec=dset.get("preprocess_spec"),
    )

    y_te   = result["y_test"]
    prob_c = result["prob"]
    pred_c = result["pred"]
    g_te   = result["g_test"]
    metrics_a = result["subgroup_metrics"]["A"]
    metrics_b = result["subgroup_metrics"]["B"]

    boot_sens  = result["bootstrap_raw"]["sensitivity"]
    boot_ppv   = result["bootstrap_raw"]["ppv"]
    boot_fnr   = result["bootstrap_raw"]["fnr"]
    boot_ppr   = result["bootstrap_raw"]["predicted_positive_rate"]
    boot_ppv_a = result["bootstrap_prevalence_adjusted"]["ppv"]

    comparison = f"{ga}_vs_{gb}"
    note = ""
    if is_expl:
        note = (
            f"NHANES exploratory: test positives {ga}={int(metrics_a['n_pos'])}; "
            f"{gb}={int(metrics_b['n_pos'])}"
        )

    raw_row = {
        "dataset": name, "comparison": comparison,
        "group_a": ga, "group_b": gb,
        "n_a": metrics_a["n"], "n_b": metrics_b["n"],
        "n_pos_a": metrics_a["n_pos"], "n_pos_b": metrics_b["n_pos"],
        "prev_a": metrics_a["prevalence"], "prev_b": metrics_b["prevalence"],
        "auroc": result["test_auroc"][result["best_model_name"]],
        "best_model": result["best_model_name"],
        "sensitivity_a": metrics_a["sensitivity"], "sensitivity_b": metrics_b["sensitivity"],
        "sensitivity_gap": boot_sens["point"],
        "sensitivity_ci_low": boot_sens["ci_low"], "sensitivity_ci_high": boot_sens["ci_high"],
        "fnr_a": metrics_a["fnr"], "fnr_b": metrics_b["fnr"],
        "fnr_gap": boot_fnr["point"],
        "fnr_ci_low": boot_fnr["ci_low"], "fnr_ci_high": boot_fnr["ci_high"],
        "ppv_raw_a": metrics_a["ppv"], "ppv_raw_b": metrics_b["ppv"],
        "ppv_gap_raw": boot_ppv["point"],
        "ppv_gap_raw_ci_low": boot_ppv["ci_low"], "ppv_gap_raw_ci_high": boot_ppv["ci_high"],
        "npv_a": metrics_a["npv"], "npv_b": metrics_b["npv"],
        "ppr_a": metrics_a["predicted_positive_rate"], "ppr_b": metrics_b["predicted_positive_rate"],
        "ppr_gap": boot_ppr["point"],
        "ppr_ci_low": boot_ppr["ci_low"], "ppr_ci_high": boot_ppr["ci_high"],
        "ppv_gap_adj": boot_ppv_a["point"],
        "ppv_gap_adj_ci_low": boot_ppv_a["ci_low"], "ppv_gap_adj_ci_high": boot_ppv_a["ci_high"],
        "ppv_attenuation_pct": result["prevalence_adjustment"]["ppv"]["attenuation_pct"],
        "exploratory": is_expl,
        "is_secondary": is_secondary,
        "note": note,
    }

    boot_rows = []
    for mk, br in result["bootstrap_raw"].items():
        boot_rows.append({"dataset": name, "comparison": comparison, "metric": mk,
                          "estimate": br["point"], "ci_low": br["ci_low"],
                          "ci_high": br["ci_high"], "type": "raw"})
    for mk, br in result["bootstrap_prevalence_adjusted"].items():
        boot_rows.append({"dataset": name, "comparison": comparison, "metric": mk,
                          "estimate": br["point"], "ci_low": br["ci_low"],
                          "ci_high": br["ci_high"], "type": "prevalence_adjusted"})

    shap_df = shapley_decompose_gap(metrics_a, metrics_b)
    shap_df.insert(0, "dataset", name)
    shap_df.insert(1, "comparison", comparison)

    cm_df = None
    if result.get("case_mix") is not None:
        cm_df = result["case_mix"].copy()
        cm_df.insert(0, "dataset", name)
        cm_df.insert(1, "comparison", comparison)
    else:
        raise RuntimeError(f"{name} [{comparison}]: required case-mix result was not produced")

    ipw_df = None
    target_prev = result["prevalence_adjustment"]["target_prevalence"]
    try:
        ipw_res = empirical_prevalence_match(
            y_te, prob_c, pred_c, g_te, ga, gb,
            target_prevalence=target_prev,
            n_boot=0 if JAMA_ONLY else 500, seed=SEED)
        ipw_df = ipw_res["table"].copy()
        ipw_df.insert(0, "dataset", name)
        ipw_df.insert(1, "comparison", comparison)
    except Exception as e:
        raise RuntimeError(f"{name} [{comparison}]: required IPW result failed: {e}") from e

    # meta SE via _ci_to_se (CONSISTENT with sex/age)
    sens_se = float(_ci_to_se(boot_sens["ci_low"], boot_sens["ci_high"]))
    meta_input = (name, comparison, boot_sens["point"], sens_se, is_expl,
                  metrics_a["n_pos"], metrics_b["n_pos"])

    return {
        "result": result, "raw_row": raw_row, "boot_rows": boot_rows,
        "shap_df": shap_df, "cm_df": cm_df, "ipw_df": ipw_df,
        "meta_input": meta_input,
        "y_te": y_te, "prob_c": prob_c, "pred_c": pred_c, "g_te": g_te,
        "boot_sens": boot_sens, "boot_ppv": boot_ppv, "boot_ppv_a": boot_ppv_a,
        "metrics_a": metrics_a, "metrics_b": metrics_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = [
    ("data/diabetes_130_hospitals.csv", _prepare_diabetes130),
    ("data/brfss2022_subset.csv",       _prepare_brfss),
    ("data/nhanes_2017_2018.csv",       _prepare_nhanes),
]

EXCLUDED = [
    ("CDC Diabetes (BRFSS 2015)", "no race/ethnicity variable in the dataset"),
    ("CCHS 2019-20",              "PUMF subset carries no race/ethnicity variable"),
]

raw_gap_rows   = []
boot_rows      = []
shapley_rows   = []
casemix_rows   = []
ipw_rows       = []
all_groups_rows = []
secondary_rows = []
stability_rows = []
meta_inputs    = []   # primary only: (name, comparison, est, se, is_expl, npos_a, npos_b)
completed_comparisons = []
run_errors = []

print("\n" + "#"*70)
print("# RACE DECOMPOSITION")
print("# Excluded datasets (no usable race column):")
for nm, why in EXCLUDED:
    print(f"#   - {nm}: {why}")
print("#"*70)

for path, loader_fn in LOADERS:
    print(f"\n{'='*70}\n  {path}\n{'='*70}")
    try:
        dset = loader_fn(path)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        run_errors.append(f"{path}: load error: {e}")
        import traceback; traceback.print_exc()
        continue

    name = dset["name"]
    df   = dset["df"]
    tc   = dset["target_col"]
    gc   = dset["group_col"]
    ga   = dset["group_a"]
    gb   = dset["group_b"]
    fc   = dset["feature_cols"]

    # subgroup sizes
    for grp in [ga, gb]:
        sub = df[df[gc] == grp]
        n = len(sub); n_pos = int(sub[tc].sum())
        print(f"  Group '{grp}': n={n:,}, positives={n_pos:,} (prev={n_pos/n:.3f})")

    # ── primary comparison ────────────────────────────────────────────────
    try:
        art = run_one_comparison(dset, ga, gb, is_secondary=False)
    except Exception as e:
        print(f"  PIPELINE ERROR (primary): {e}")
        run_errors.append(f"{path}: primary comparison error: {e}")
        import traceback; traceback.print_exc()
        continue

    raw_gap_rows.append(art["raw_row"])
    boot_rows.extend(art["boot_rows"])
    shapley_rows.append(art["shap_df"])
    if art["cm_df"] is not None:
        casemix_rows.append(art["cm_df"])
    if art["ipw_df"] is not None:
        ipw_rows.append(art["ipw_df"])
    meta_inputs.append(art["meta_input"])
    completed_comparisons.append((name, f"{ga}_vs_{gb}"))

    # ── automatic 10-seed stability check if any group < 100 test positives ─
    npos_a = art["metrics_a"]["n_pos"]
    npos_b = art["metrics_b"]["n_pos"]
    primary_stability_key = (name, ga, gb)
    primary_small = min(npos_a, npos_b) < STABILITY_POS_THRESHOLD
    primary_forced = primary_stability_key in FORCED_STABILITY_COMPARISONS
    if not AUXILIARY_ONLY and (primary_small or primary_forced):
        trigger_reason = ("small_test_positive_count" if primary_small
                          else "prespecified_follow_up")
        print(f"\n  [stability] {name}: min test positives = {min(npos_a, npos_b)} "
              f"({trigger_reason}) -> running legacy fixed-logistic 10-seed direction check")
        chk = stability_check_10seed(
            df, fc, tc, gc, ga, gb,
            preprocess_spec=dset.get("preprocess_spec"),
            cluster_ids=dset.get("cluster_ids"),
        )
        chk_row = {"dataset": name, "comparison": f"{ga}_vs_{gb}",
                   "trigger_min_test_pos": int(min(npos_a, npos_b)),
                   "trigger_reason": trigger_reason, **chk}
        stability_rows.append(chk_row)
        print(f"    mean gap={chk['mean_gap']:+.4f} (sd={chk['sd_gap']:.4f}), "
              f"range=[{chk['min_gap']:+.4f}, {chk['max_gap']:+.4f}], "
              f"{chk['n_seeds_majority']}/10 {chk['majority_direction']}, "
              f"stable={chk['stable']}")

    # ── all-groups metrics ────────────────────────────────────────────────
    try:
        work_all = df[df[gc].isin(dset["all_groups"])].copy()
        primary_train_index = pd.Index(art["result"]["train_index"])
        primary_test_index = pd.Index(art["result"]["test_index"])
        nonprimary_groups = [
            group for group in dset["all_groups"] if group not in (ga, gb)
        ]
        # With a clustering identifier, row-level disjointness is not enough:
        # a non-primary-group encounter can belong to a patient who trained
        # the model. Restrict the pool to patients unseen in training.
        cluster_ids = dset.get("cluster_ids")
        if cluster_ids is not None:
            training_clusters = set(cluster_ids.reindex(primary_train_index))
            pool_clusters = cluster_ids.reindex(work_all.index)
            nonprimary_pool = work_all[~pool_clusters.isin(training_clusters)]
        else:
            nonprimary_pool = work_all

        nonprimary_test_index = _nonprimary_test_indices(
            nonprimary_pool, tc, gc, nonprimary_groups,
            test_size=0.3, random_state=SEED,
        )
        evaluation_index = primary_test_index.append(nonprimary_test_index)
        if evaluation_index.has_duplicates:
            raise RuntimeError(f"{name}: duplicate rows in all-groups evaluation set")
        assert_no_training_overlap(primary_train_index, evaluation_index, context=name)
        if cluster_ids is not None:
            assert_no_training_cluster_overlap(
                cluster_ids.reindex(primary_train_index),
                cluster_ids.reindex(evaluation_index),
                context=f"{name} all-groups",
            )

        evaluation = work_all.loc[evaluation_index]
        # Apply the SAME training-fitted preprocessing used to fit the model.
        X_te_all = art["result"]["preprocessor"].transform(
            evaluation, partition="all_groups_auxiliary"
        )
        y_te_all = evaluation[tc].astype(int)
        g_te_all = evaluation[gc]
        calib_model = art["result"]["calibrated_model"]
        thr_all = float(art["result"]["decision_threshold"])
        prob_all = calib_model.predict_proba(X_te_all)[:, 1]
        pred_all = (prob_all >= thr_all).astype(int)
        ab = _all_groups_metrics(y_te_all.values, prob_all, pred_all,
                                 g_te_all.values, dset["all_groups"], name,
                                 primary_groups=(ga, gb),
                                 best_model=art["result"]["best_model_name"],
                                 decision_threshold=thr_all)
        if (len(ab) != len(dset["all_groups"])
                or int(ab["training_overlap_n"].sum()) != 0):
            raise RuntimeError(f"{name}: incomplete or contaminated all-groups output")
        all_groups_rows.append(ab)
    except Exception as e:
        raise RuntimeError(f"{name}: required all-groups metrics failed: {e}") from e

    # ── secondary comparisons (BRFSS) ─────────────────────────────────────
    secondary_comparisons = [] if AUXILIARY_ONLY else dset["secondary"]
    for (sa, sb) in secondary_comparisons:
        print(f"\n  --- secondary: {sa} vs {sb} ---")
        try:
            art_s = run_one_comparison(dset, sa, sb, is_secondary=True)
            secondary_rows.append(art_s["raw_row"])
            boot_rows.extend(art_s["boot_rows"])
            shapley_rows.append(art_s["shap_df"])
            if art_s["cm_df"] is not None:
                casemix_rows.append(art_s["cm_df"])
            if art_s["ipw_df"] is not None:
                ipw_rows.append(art_s["ipw_df"])
            completed_comparisons.append((name, f"{sa}_vs_{sb}"))
            # Stability check for a small subgroup or a prespecified fragile
            # secondary finding whose evidence must remain reproducible.
            npa = art_s["metrics_a"]["n_pos"]; npb = art_s["metrics_b"]["n_pos"]
            secondary_stability_key = (name, sa, sb)
            secondary_small = min(npa, npb) < STABILITY_POS_THRESHOLD
            secondary_forced = secondary_stability_key in FORCED_STABILITY_COMPARISONS
            if secondary_small or secondary_forced:
                trigger_reason = ("small_test_positive_count" if secondary_small
                                  else "prespecified_follow_up")
                print(f"\n  [stability] {name} [{sa} vs {sb}]: "
                      f"min test positives = {min(npa, npb)} ({trigger_reason}) "
                      "-> running legacy fixed-logistic 10-seed direction check")
                chk = stability_check_10seed(df, fc, tc, gc, sa, sb)
                stability_rows.append({"dataset": name, "comparison": f"{sa}_vs_{sb}",
                                       "trigger_min_test_pos": int(min(npa, npb)),
                                       "trigger_reason": trigger_reason, **chk})
        except Exception as e:
            print(f"  PIPELINE ERROR (secondary {sa} vs {sb}): {e}")
            run_errors.append(f"{path}: secondary {sa} vs {sb} error: {e}")
            import traceback; traceback.print_exc()

    print(f"\n  Sensitivity gap ({ga} - {gb}): {art['boot_sens']['point']:+.4f} "
          f"(95% CI [{art['boot_sens']['ci_low']:+.4f}, {art['boot_sens']['ci_high']:+.4f}])")
    print(f"  PPV gap raw={art['boot_ppv']['point']:+.4f}, adj={art['boot_ppv_a']['point']:+.4f}, "
          f"attenuation={art['result']['prevalence_adjustment']['ppv']['attenuation_pct']:.1f}%")
    if dset["exploratory"]:
        print("  *** EXPLORATORY: marginal subgroup sizes ***")


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary-only production mode replaces only race_all_groups.csv after all
# three primary datasets succeed. No primary, secondary, stability, or meta file
# is touched.
if AUXILIARY_ONLY:
    expected_primary = len(LOADERS)
    if (run_errors or len(all_groups_rows) != expected_primary
            or len(raw_gap_rows) != expected_primary):
        raise RuntimeError(
            "Race auxiliary rerun incomplete; no result files were written. "
            f"Primary results={len(raw_gap_rows)}/{expected_primary}; "
            f"all-groups tables={len(all_groups_rows)}; errors={run_errors}"
        )
    auxiliary = pd.concat(all_groups_rows, ignore_index=True)
    auxiliary_path = os.path.join(OUT_DIR, "race_all_groups.csv")
    _atomic_write_csv(auxiliary, auxiliary_path)
    print(f"\n[OK] Saved only {auxiliary_path}")
    raise SystemExit(0)


# Refuse to write a partial production result set. Every primary comparison,
# secondary comparison, and required downstream component must exist before
# the first CSV is replaced.
expected_primary = len(LOADERS)
expected_secondary = EXPECTED_SECONDARY_COMPARISONS
expected_total = expected_primary + expected_secondary
component_counts = {
    "primary raw gaps": len(raw_gap_rows),
    "secondary raw gaps": len(secondary_rows),
    "bootstrap rows": len(boot_rows),
    "Shapley comparisons": len(shapley_rows),
    "case-mix comparisons": len(casemix_rows),
    "IPW comparisons": len(ipw_rows),
    "all-groups datasets": len(all_groups_rows),
    "meta inputs": len(meta_inputs),
    "completed comparisons": len(completed_comparisons),
    "stability checks": len(stability_rows),
}
expected_counts = {
    "primary raw gaps": expected_primary,
    "secondary raw gaps": expected_secondary,
    "bootstrap rows": expected_total * 10,
    "Shapley comparisons": expected_total,
    "case-mix comparisons": expected_total,
    "IPW comparisons": expected_total,
    "all-groups datasets": expected_primary,
    "meta inputs": expected_primary,
    "completed comparisons": expected_total,
    "stability checks": 2,
}
incomplete = {
    key: {"actual": component_counts[key], "expected": expected}
    for key, expected in expected_counts.items()
    if component_counts[key] != expected
}
expected_stability = {
    ("NHANES 2017-18 (diabetes)", "Black_vs_White"),
    ("BRFSS 2022 (heart disease)", "Hispanic_vs_White"),
}
actual_stability = {(row["dataset"], row["comparison"]) for row in stability_rows}
if run_errors or incomplete or actual_stability != expected_stability:
    details = "; ".join(run_errors) if run_errors else "no explicit comparison error recorded"
    raise RuntimeError(
        "Race rerun incomplete; no result files were written. "
        f"component counts={component_counts}; incomplete={incomplete}; "
        f"stability checks={sorted(actual_stability)}; errors={details}"
    )


# Save per-dataset tables
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(raw_gap_rows).to_csv(os.path.join(OUT_DIR, "race_raw_gaps.csv"), index=False)
print("\nSaved race_raw_gaps.csv")
pd.DataFrame(boot_rows).to_csv(os.path.join(OUT_DIR, "race_bootstrap.csv"), index=False)
print("Saved race_bootstrap.csv")
if shapley_rows and not JAMA_ONLY:
    pd.concat(shapley_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "race_shapley.csv"), index=False)
    print("Saved race_shapley.csv")
if casemix_rows:
    pd.concat(casemix_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "race_casemix_waterfall.csv"), index=False)
    print("Saved race_casemix_waterfall.csv")
if ipw_rows:
    pd.concat(ipw_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "race_ipw_adjustment.csv"), index=False)
    print("Saved race_ipw_adjustment.csv")
if all_groups_rows and not JAMA_ONLY:
    pd.concat(all_groups_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "race_all_groups.csv"), index=False)
    print("Saved race_all_groups.csv")
if secondary_rows:
    pd.DataFrame(secondary_rows).to_csv(os.path.join(OUT_DIR, "race_secondary_comparisons.csv"), index=False)
    print("Saved race_secondary_comparisons.csv")
if stability_rows:
    pd.DataFrame(stability_rows).to_csv(os.path.join(OUT_DIR, "race_stability_checks.csv"), index=False)
    print("Saved race_stability_checks.csv")

if JAMA_ONLY:
    print("\n[OK] Saved JAMA-scoped race outputs only:", OUT_DIR)
    raise SystemExit(0)


# ─────────────────────────────────────────────────────────────────────────────
# DerSimonian-Laird meta-analysis of sensitivity gap (primary comparisons)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DerSimonian-Laird meta-analysis: sensitivity gap (minority vs reference)")
print("SE via _ci_to_se (consistent with sex + age analyses)")
print("="*70)

names_all = [f"{t[0]} [{t[1]}]" for t in meta_inputs]
ests_all  = np.array([t[2] for t in meta_inputs])
ses_all   = np.array([t[3] for t in meta_inputs])

summary_all, per_study_all = _dl_meta(names_all, ests_all, ses_all)
print(f"All {len(meta_inputs)} datasets:")
print(f"  Pooled = {summary_all['pooled_estimate']:+.4f} "
      f"(95% CI [{summary_all['pooled_ci_low']:+.4f}, {summary_all['pooled_ci_high']:+.4f}])")
print(f"  PI = [{summary_all['prediction_interval_low']}, {summary_all['prediction_interval_high']}]")
print(f"  I^2 = {summary_all['I2_pct']:.1f}%  tau^2 = {summary_all['tau2']:.5f}  "
      f"Q={summary_all['Q']:.3f} (df={summary_all['Q_df']}, p={summary_all['Q_pvalue']:.4f})")

# excluding NHANES (exploratory)
non_expl = [t for t in meta_inputs if not t[4]]
summary_ne = None
per_study_ne = None
if len(non_expl) >= 2:
    names_ne = [f"{t[0]} [{t[1]}]" for t in non_expl]
    ests_ne  = np.array([t[2] for t in non_expl])
    ses_ne   = np.array([t[3] for t in non_expl])
    summary_ne, per_study_ne = _dl_meta(names_ne, ests_ne, ses_ne)
    print(f"\nExcluding NHANES ({len(non_expl)} datasets):")
    print(f"  Pooled = {summary_ne['pooled_estimate']:+.4f} "
          f"(95% CI [{summary_ne['pooled_ci_low']:+.4f}, {summary_ne['pooled_ci_high']:+.4f}])")
    print(f"  I^2 = {summary_ne['I2_pct']:.1f}%  tau^2 = {summary_ne['tau2']:.5f}")

meta_summary_rows = [{"analysis": "all_3_datasets", **summary_all}]
if summary_ne is not None:
    meta_summary_rows.append({"analysis": "excl_nhanes_exploratory", **summary_ne})
pd.DataFrame(meta_summary_rows).to_csv(
    os.path.join(OUT_DIR, "race_meta_analysis_sensitivity_gap.csv"), index=False)
print("\nSaved race_meta_analysis_sensitivity_gap.csv")

per_study_all["analysis"] = "all_3_datasets"
if per_study_ne is not None:
    per_study_ne["analysis"] = "excl_nhanes_exploratory"
    per_study_combined = pd.concat([per_study_all, per_study_ne], ignore_index=True)
else:
    per_study_combined = per_study_all
per_study_combined.to_csv(os.path.join(OUT_DIR, "race_meta_analysis_per_study.csv"), index=False)
print("Saved race_meta_analysis_per_study.csv")

print("\n[OK] All race decomposition outputs saved to:", OUT_DIR)
