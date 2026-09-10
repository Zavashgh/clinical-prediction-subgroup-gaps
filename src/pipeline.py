"""
pipeline.py
===========
The single reusable entry point for the whole project:

    run_fairness_analysis(df, feature_cols, target_col, group_col,
                           group_a_value, group_b_value, group_labels, ...)

Every dataset (CDC Diabetes, Diabetes 130-US Hospitals, BRFSS, NHANES,
CCHS, ...) is loaded by a dataset-specific function in `datasets.py` and
then passed through this *same* function, so results are directly
comparable across datasets.

Pipeline steps (see README for the full methodological writeup):
  1. Stratified train/test split (fixed seed).
  2. Compare the required six candidate model families using five-fold
     cross-validation on the training data only; fail closed if any required
     family cannot be constructed.
  3. Fit the eligible family with the highest training-CV AUROC, calibrate it
     with five-fold isotonic fold ensembling on training data only, and touch
     the held-out test set exactly once for final evaluation.
  4. Compute raw per-subgroup metrics (discrimination, threshold, fairness,
     calibration) for the calibrated model.
  5. Prevalence adjustment: recompute PPV/NPV/predicted-positive-rate gaps
     as if both groups had the same disease prevalence.
  6. Exploratory calibration/threshold analysis: compare a shared threshold
     vs. test-derived thresholds that equalize sensitivity across groups.
  7. Case-mix waterfall: how much of the *outcome* gap is explained by each
     block of covariates.
  8. Bootstrap confidence intervals for every gap (raw and adjusted).
"""

from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
import sys

from .runtime import (
    DETERMINISTIC_N_JOBS,
    configure_deterministic_environment,
    enforce_loaded_threadpool_limits,
    runtime_provenance,
)

_DETERMINISTIC_ENVIRONMENT = configure_deterministic_environment()

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from . import metrics as fm
from . import adjustments as adj
from . import uncertainty as unc
from .preprocessing import TrainFittedPreprocessor

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
    _XGB_IMPORT_ERROR = None
except ImportError as exc:
    _HAS_XGB = False
    _XGB_IMPORT_ERROR = exc

_THREADPOOL_LIMITER = enforce_loaded_threadpool_limits()


CALIBRATION_PROTOCOL = "five_fold_isotonic_fold_ensemble"
GROUPED_CALIBRATION_PROTOCOL = "five_fold_isotonic_fold_ensemble_patient_grouped"
SPLIT_PROTOCOL_ROW = "stratified_row_level_train_test_split"
SPLIT_PROTOCOL_GROUPED = "patient_grouped_shuffle_split_unstratified"
BOOTSTRAP_PROTOCOL_ROW = "row_level_within_subgroup_resampling"
BOOTSTRAP_PROTOCOL_CLUSTER = "patient_cluster_within_subgroup_resampling"
PRIMARY_THRESHOLD_PROTOCOL = "training_outcome_prevalence"
CASE_MIX_INFERENCE_PROTOCOL = "exploratory_point_estimates_no_confidence_intervals"
REQUIRED_MODEL_FAMILIES = (
    "logistic_regression",
    "l1_logistic_regression",
    "random_forest",
    "shallow_tree",
    "xgboost",
    "calibrated_ensemble",
)
_PACKAGE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "matplotlib",
    "xgboost",
    "pyreadstat",
)
_PACKAGE_VERSION_CACHE = None


def _json_safe(value):
    """Convert NumPy/nonfinite values into strict JSON-compatible objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _package_versions():
    global _PACKAGE_VERSION_CACHE
    if _PACKAGE_VERSION_CACHE is None:
        versions = {}
        for distribution in _PACKAGE_DISTRIBUTIONS:
            try:
                versions[distribution] = importlib_metadata.version(distribution)
            except importlib_metadata.PackageNotFoundError:
                versions[distribution] = None
        _PACKAGE_VERSION_CACHE = versions
    return dict(_PACKAGE_VERSION_CACHE)


def _persist_pipeline_provenance(
    result,
    *,
    target_col,
    group_col,
    group_a_value,
    group_b_value,
    label_a,
    label_b,
    random_state,
    n_rows,
    n_features,
    analysis_label,
):
    """Append one strict-JSON analytical record when production capture is enabled."""
    destination = os.environ.get("MEDICAL_FAIRNESS_PROVENANCE_JSONL")
    if not destination:
        return None

    input_hashes_raw = os.environ.get("MEDICAL_FAIRNESS_INPUT_HASHES_JSON", "[]")
    try:
        input_hashes = json.loads(input_hashes_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "MEDICAL_FAIRNESS_INPUT_HASHES_JSON is not valid JSON"
        ) from exc

    record = {
        "record_type": "pipeline_analysis",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_id": os.environ.get("MEDICAL_FAIRNESS_COMMAND_ID"),
        "git": {
            "commit": os.environ.get("MEDICAL_FAIRNESS_GIT_COMMIT"),
            "branch": os.environ.get("MEDICAL_FAIRNESS_GIT_BRANCH"),
            "source_dirty": os.environ.get("MEDICAL_FAIRNESS_SOURCE_DIRTY") == "1",
        },
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "packages": _package_versions(),
        },
        "inputs": input_hashes,
        "execution": {
            "random_seed": int(random_state),
            "runtime": result["runtime_provenance"],
        },
        "analysis": {
            "dataset": analysis_label,
            "target_col": target_col,
            "protected_attribute": group_col,
            "group_a_value": group_a_value,
            "group_b_value": group_b_value,
            "comparison": f"{label_a} vs {label_b}",
            "n_rows": int(n_rows),
            "n_features": int(n_features),
        },
        "partitioning": {
            "split_protocol": result["split_protocol"],
            "bootstrap_protocol": result["bootstrap_protocol"],
            "diagnostics": result["split_diagnostics"],
        },
        "preprocessing": result["preprocessing_schema"],
        "model_selection": {
            "candidate_families": result["candidate_model_families"],
            "selected_family": result["best_model_name"],
            "protocol": result["selection_cv_protocol"],
            "eligibility_rule": "finite mean CV AUROC > 0.5; all families must complete CV",
            "families": result["cv_auroc_details"],
        },
        "calibration": {
            "protocol": result["calibration_protocol"],
            "method": "isotonic",
            "cv_folds": 5,
            "ensemble": True,
        },
        "threshold": {
            "protocol": result["threshold_protocol"],
            "request": result["threshold_request"],
            "value": result["decision_threshold"],
            "derivation_partition": (
                "training_set" if result["threshold_protocol"] == PRIMARY_THRESHOLD_PROTOCOL
                else "explicit_nonprimary"
            ),
        },
        "final_evaluation": {
            "test_auroc": result["final_test_auroc"],
            "n_test": result["n_test"],
        },
        "reporting": {
            "group_a": {
                "label": label_a,
                "value": group_a_value,
                "counts": {
                    key: result["subgroup_metrics"]["A"][key]
                    for key in ("n", "n_pos", "n_neg", "tp", "fp", "tn", "fn")
                },
                "metrics_with_confidence_intervals": result["subgroup_metric_cis"]["A"],
            },
            "group_b": {
                "label": label_b,
                "value": group_b_value,
                "counts": {
                    key: result["subgroup_metrics"]["B"][key]
                    for key in ("n", "n_pos", "n_neg", "tp", "fp", "tn", "fn")
                },
                "metrics_with_confidence_intervals": result["subgroup_metric_cis"]["B"],
            },
        },
        "case_mix_inference": result["case_mix_inference_protocol"],
        "outputs": [],
        "outputs_pending_command_finalization": True,
    }
    record = _json_safe(record)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    with open(destination, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return record


def _build_models(random_state):
    """Candidate base model families for selection AND the robustness check.

    Single source of truth for the six families named in the Methods:
    logistic regression, L1-penalized logistic regression, random forest,
    a shallow decision tree, XGBoost, and a soft-voting ensemble of LR+RF+XGB.
    All six families are required. If XGBoost or construction of any family is
    unavailable, fail closed rather than silently evaluating a smaller pool.
    LR gets standardized features via a Pipeline. This same pool is used both
    to select the primary model (training-set CV) and by the model-family
    robustness analysis, so the pools are identical.
    """
    if not _HAS_XGB:
        raise RuntimeError(
            "Cannot construct the required six-family candidate pool: XGBoost "
            f"is unavailable ({type(_XGB_IMPORT_ERROR).__name__}: "
            f"{_XGB_IMPORT_ERROR}). No reduced four-family fallback is allowed."
        ) from _XGB_IMPORT_ERROR

    try:
        models = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, C=1.0, random_state=random_state,
                )),
            ]),
            "l1_logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, l1_ratio=1.0, solver="liblinear",
                    C=0.5, random_state=random_state,
                )),
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                n_jobs=DETERMINISTIC_N_JOBS, random_state=random_state,
            ),
            "shallow_tree": DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=20, random_state=random_state,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=random_state,
                n_jobs=DETERMINISTIC_N_JOBS, device="cpu",
            ),
            "calibrated_ensemble": VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(
                        max_iter=2000, random_state=random_state,
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=200, min_samples_leaf=5,
                        n_jobs=DETERMINISTIC_N_JOBS, random_state=random_state,
                    )),
                    ("xgb", XGBClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="logloss", random_state=random_state,
                        n_jobs=DETERMINISTIC_N_JOBS, device="cpu",
                    )),
                ],
                voting="soft",
                n_jobs=DETERMINISTIC_N_JOBS,
            ),
        }
    except Exception as exc:
        raise RuntimeError(
            "Cannot construct the required six-family candidate pool: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    missing = [name for name in REQUIRED_MODEL_FAMILIES if name not in models]
    if missing or len(models) != len(REQUIRED_MODEL_FAMILIES):
        raise RuntimeError(
            "Required six-family candidate pool is incomplete. "
            f"Expected={list(REQUIRED_MODEL_FAMILIES)!r}; "
            f"constructed={list(models)!r}; missing={missing!r}."
        )
    return models


def _metric_counts(metrics, key):
    """Numerator and denominator behind a threshold metric, for reporting."""
    table = {
        "prevalence": (metrics["n_pos"], metrics["n"]),
        "sensitivity": (metrics["tp"], metrics["n_pos"]),
        "specificity": (metrics["tn"], metrics["n_neg"]),
        "ppv": (metrics["tp"], metrics["tp"] + metrics["fp"]),
        "npv": (metrics["tn"], metrics["tn"] + metrics["fn"]),
    }
    numerator, denominator = table[key]
    return int(numerator), int(denominator)


def _model_pipeline(preprocessor, estimator):
    """Bind data-derived preprocessing to an estimator as one fittable unit.

    Returning a Pipeline is what makes fold isolation automatic: every
    ``cross_val_score`` fold and every ``CalibratedClassifierCV`` fold clones
    this object and calls ``fit`` on its own training slice, so the medians and
    categorical schema are relearned per fold. Passing an already-transformed
    matrix instead would reuse one outer-training preprocessor inside every
    fold, which is exactly the leakage this design prevents.

    The preprocessor is cloned so the caller's outer-training instance is never
    mutated by fold fitting.
    """
    return Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("estimator", estimator),
    ])


def _make_calibrated_classifier(estimator, cv=5):
    """Build the project's explicit isotonic fold-ensemble calibrator.

    ``ensemble=True`` fits one estimator/calibrator pair per cross-validation
    fold and averages their predicted probabilities. Setting this explicitly
    preserves the project's previously approved behavior and avoids relying on
    scikit-learn's version-dependent ``ensemble="auto"`` default.
    """
    return CalibratedClassifierCV(
        estimator,
        method="isotonic",
        cv=cv,
        n_jobs=DETERMINISTIC_N_JOBS,
        ensemble=True,
    )


def _assert_no_cluster_leakage_in_folds(folds, cluster_array, context):
    """Fail closed if any cluster spans a fold's train and validation halves."""
    for fold_number, (train_positions, valid_positions) in enumerate(folds, 1):
        shared = set(cluster_array[train_positions]) & set(cluster_array[valid_positions])
        if shared:
            raise RuntimeError(
                f"{context} fold {fold_number}: {len(shared)} cluster(s) appear in "
                "both the fold-training and fold-validation halves"
            )
    return True


def _select_best_model(cv_auroc, cv_errors=None):
    """Return the best eligible model family or raise a diagnostic error.

    A family is eligible only when its mean training-CV AUROC is finite and
    strictly greater than 0.5. Degenerate or failed families are never
    re-admitted as a fallback.
    """
    cv_errors = cv_errors or {}
    eligible = {
        name: float(score)
        for name, score in cv_auroc.items()
        if np.isfinite(score) and float(score) > 0.5
    }
    if eligible:
        return max(eligible, key=eligible.get)

    diagnostics = []
    for name in cv_auroc:
        score = cv_auroc[name]
        if name in cv_errors:
            status = f"failed: {cv_errors[name]}"
        elif not np.isfinite(score):
            status = f"invalid/nonfinite AUROC: {score!r}"
        else:
            status = f"ineligible AUROC <= 0.5: {float(score):.12g}"
        diagnostics.append(f"{name} [{status}]")
    if not diagnostics:
        diagnostics.append("no candidate families were supplied")
    raise RuntimeError(
        "Model selection failed: no candidate family had a finite training-CV "
        "AUROC strictly greater than 0.5. Per-family diagnostics: "
        + "; ".join(diagnostics)
    )


def _require_complete_cv_evaluation(candidate_names, cv_auroc, cv_errors=None):
    """Fail closed if any constructed candidate lacks a valid CV result."""
    cv_errors = cv_errors or {}
    missing = [name for name in candidate_names if name not in cv_auroc]
    invalid = [
        name for name in candidate_names
        if name in cv_auroc and not np.isfinite(cv_auroc[name])
    ]
    failed = [name for name in candidate_names if name in cv_errors]
    if not (missing or invalid or failed):
        return

    details = []
    for name in candidate_names:
        if name in cv_errors:
            details.append(f"{name} [failed: {cv_errors[name]}]")
        elif name in missing:
            details.append(f"{name} [missing CV result]")
        elif name in invalid:
            details.append(f"{name} [nonfinite CV AUROC: {cv_auroc[name]!r}]")
    raise RuntimeError(
        "Model selection aborted: the required candidate pool was not fully "
        "available during five-fold training CV. " + "; ".join(details)
    )


def run_fairness_analysis(
    df,
    feature_cols,
    target_col,
    group_col,
    group_a_value,
    group_b_value,
    group_labels=None,
    threshold="prevalence",
    test_size=0.3,
    random_state=42,
    n_boot=1000,
    covariate_blocks=None,
    run_calibration_adjustment=True,
    run_case_mix=True,
    allow_nonprimary_threshold=False,
    verbose=True,
    analysis_label=None,
    cluster_ids=None,
    preprocess_spec=None,
):
    """
    Run the full subgroup-fairness pipeline on one dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset, one row per patient.
    feature_cols : list of str
        Columns to use as model features (should NOT include target_col or
        group_col).
    target_col : str
        Binary outcome column (1 = event present).
    group_col : str
        Protected-attribute column (e.g. "Sex").
    group_a_value, group_b_value : scalar
        Values of `group_col` defining the two subgroups compared. All
        gaps are reported as (group A) - (group B).
    group_labels : dict, optional
        {raw value: display label}, used only for printing.
    threshold : "prevalence" or float
        The approved primary rule is "prevalence": the unweighted outcome
        prevalence in the TRAINING set after filtering to the compared groups.
        A numeric threshold is rejected unless `allow_nonprimary_threshold`
        is explicitly true for a robustness/exploratory analysis.
    test_size, random_state : split parameters.
    n_boot : int
        Number of bootstrap resamples for confidence intervals.
    covariate_blocks : dict, optional
        {block name: [columns]} for the case-mix waterfall.
    run_calibration_adjustment, run_case_mix : bool
        Toggle the heavier optional steps.
    allow_nonprimary_threshold : bool
        Permit a numeric threshold only for explicitly non-primary analyses.
    verbose : bool
        Print human-readable summary tables as the pipeline runs.
    analysis_label : str, optional
        Human-readable dataset label persisted in production provenance. It
        has no effect on model fitting or evaluation.

    Returns
    -------
    dict with keys: "models", "best_model_name", "test_auroc",
    "subgroup_metrics", "fairness_gaps", "prevalence_adjustment",
    "bootstrap", "calibration_adjustment" (optional exploratory/descriptive
    analysis),
    "case_mix" (optional), "group_labels", "n_test".
    """
    label_a = (group_labels or {}).get(group_a_value, str(group_a_value))
    label_b = (group_labels or {}).get(group_b_value, str(group_b_value))

    # ------------------------------------------------------------------
    # 1. Restrict to the compared groups, then split.
    #
    #    Order matters. The restriction to the two compared groups and any
    #    comparison-specific predictor exclusions are applied by the caller
    #    BEFORE this point, so the training-fitted preprocessing below is
    #    learned from exactly the rows and columns this comparison uses --
    #    never from groups excluded from the comparison.
    #
    #    When `cluster_ids` is supplied (Diabetes-130 patients), the split is
    #    group-aware: no cluster may appear on both sides. scikit-learn 1.9
    #    offers no simultaneously stratified and grouped shuffle-split, so the
    #    grouped path uses GroupShuffleSplit, which preserves the exact 70/30
    #    size concept and seed but not outcome stratification. The realized
    #    class proportions are recorded in `split_diagnostics` rather than
    #    assumed.
    # ------------------------------------------------------------------
    work = df[df[group_col].isin([group_a_value, group_b_value])].copy()
    y = work[target_col].astype(int)
    group = work[group_col]

    clusters = None
    if cluster_ids is not None:
        if isinstance(cluster_ids, pd.Series):
            # Align by index so callers that filtered rows upstream stay correct.
            clusters = cluster_ids.reindex(work.index)
        else:
            clusters = pd.Series(np.asarray(cluster_ids), index=df.index).loc[work.index]
        if clusters.isnull().any():
            raise ValueError(
                "cluster_ids is missing or unaligned for some compared rows"
            )

    if clusters is None:
        train_index, test_index = train_test_split(
            work.index, test_size=test_size, random_state=random_state, stratify=y
        )
        split_protocol = SPLIT_PROTOCOL_ROW
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        positions_train, positions_test = next(
            splitter.split(work.index, y, groups=clusters.to_numpy())
        )
        train_index = work.index[positions_train]
        test_index = work.index[positions_test]
        split_protocol = SPLIT_PROTOCOL_GROUPED

    train_index = pd.Index(train_index)
    test_index = pd.Index(test_index)

    # ---- Training-fitted preprocessing (fit on training rows ONLY) -------
    spec = preprocess_spec or {"impute_cols": [], "categorical": []}
    preprocessor = TrainFittedPreprocessor(
        feature_order=feature_cols,
        categorical_specs=spec.get("categorical", []),
    )
    # The OUTER-training preprocessor. It is fitted on the outer-training rows
    # and is what the final model and the held-out evaluation use.
    #
    # It is deliberately NOT the object used inside cross-validation or
    # calibration. Those folds each fit their own preprocessor on their own
    # training slice, via the pipeline built in `_model_pipeline` below, so an
    # inner validation row never contributes to the medians or categorical
    # schema of the model that is then evaluated on it.
    preprocessor.fit(work.loc[train_index])
    raw_train = work.loc[train_index]
    raw_test = work.loc[test_index]
    X_train = preprocessor.transform(raw_train, partition="train")
    X_test = preprocessor.transform(raw_test, partition="test")
    design_feature_names = list(preprocessor.feature_names_)

    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    g_train = group.loc[train_index]
    g_test = group.loc[test_index]
    y_test_arr = y_test.values
    g_test_arr = g_test.values

    clusters_train = clusters_test = None
    if clusters is not None:
        clusters_train = clusters.loc[train_index]
        clusters_test = clusters.loc[test_index]
        overlap = set(clusters_train.unique()) & set(clusters_test.unique())
        if overlap:
            raise RuntimeError(
                f"Patient-grouped split failed: {len(overlap)} cluster(s) appear "
                "in both the training and test partitions"
            )

    split_diagnostics = {
        "split_protocol": split_protocol,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "n_train_events": int(y_train.sum()),
        "n_test_events": int(y_test.sum()),
        "train_event_rate": float(y_train.mean()),
        "test_event_rate": float(y_test.mean()),
        "realized_test_fraction": float(len(test_index) / len(work)),
        "outcome_stratified": clusters is None,
        "grouped_by_cluster": clusters is not None,
    }
    if clusters is not None:
        split_diagnostics.update({
            "n_train_clusters": int(clusters_train.nunique()),
            "n_test_clusters": int(clusters_test.nunique()),
            "n_total_clusters": int(clusters.nunique()),
            "cluster_overlap_train_test": 0,
        })

    if verbose:
        print("=" * 70)
        print("STEP 1: Train/test split")
        print("=" * 70)
        print(f"Split protocol: {split_protocol}")
        print(f"Train: n={len(X_train):,}   Test: n={len(X_test):,}")
        print(f"Train event rate={y_train.mean():.4f}   "
              f"Test event rate={y_test.mean():.4f}")
        if clusters is not None:
            print(f"  Clusters: train={clusters_train.nunique():,}, "
                  f"test={clusters_test.nunique():,}, overlap=0")
        print(f"Design matrix: {len(design_feature_names)} columns "
              f"({len(preprocessor.medians_)} median-imputed on training rows)")
        for val, lbl in [(group_a_value, label_a), (group_b_value, label_b)]:
            mask = g_test_arr == val
            n = mask.sum()
            n_pos = y_test_arr[mask].sum()
            print(f"  Test subgroup '{lbl}': n={n:,}, positives={n_pos:,} "
                  f"(prevalence={n_pos / n:.3f})")
        print()

    # ------------------------------------------------------------------
    # 2 & 3. Model selection on the TRAINING data ONLY (internal k-fold
    #        cross-validation), then fit + calibrate the selected family
    #        and touch the test set exactly once for final evaluation.
    #        The test set is never used to choose the model family.
    # ------------------------------------------------------------------
    models = _build_models(random_state)
    # Group-aware selection folds when clusters exist, so one patient's
    # encounters cannot straddle a fold boundary during model-family
    # selection. StratifiedGroupKFold keeps outcome stratification here (it
    # is a k-fold, not a shuffle-split, so it is available in this version).
    if clusters is None:
        selection_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        selection_fit_params = {}
        selection_cv_protocol = "stratified_kfold_5"
    else:
        selection_cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        selection_fit_params = {"groups": clusters_train.to_numpy()}
        selection_cv_protocol = "stratified_group_kfold_5_patient_grouped"
    cv_auroc = {}
    cv_auroc_details = {}
    cv_errors = {}

    if verbose:
        print("=" * 70)
        print("STEP 2: Model selection (training-set CV AUROC; test set untouched)")
        print("=" * 70)

    for name, model in models.items():
        try:
            # Raw rows in, pipeline out: each fold clones the pipeline and
            # refits the preprocessor on that fold's training slice only.
            scores = cross_val_score(
                _model_pipeline(preprocessor, model), raw_train, y_train,
                cv=selection_cv,
                scoring="roc_auc", n_jobs=DETERMINISTIC_N_JOBS,
                **selection_fit_params,
            )
            cv_auroc[name] = float(np.mean(scores))
            cv_auroc_details[name] = {
                "status": (
                    "eligible" if np.isfinite(cv_auroc[name]) and cv_auroc[name] > 0.5
                    else "ineligible" if np.isfinite(cv_auroc[name])
                    else "invalid"
                ),
                "fold_scores": [float(value) for value in scores],
                "mean": cv_auroc[name],
                "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "error": None,
            }
        except Exception as e:
            cv_auroc[name] = float("nan")
            cv_errors[name] = f"{type(e).__name__}: {e}"
            cv_auroc_details[name] = {
                "status": "failed",
                "fold_scores": [],
                "mean": None,
                "sd": None,
                "error": cv_errors[name],
            }
            if verbose:
                print(f"  {name:22s} CV failed ({e}); excluded from selection")
        if verbose and not np.isnan(cv_auroc[name]):
            print(f"  {name:22s} train CV AUROC = {cv_auroc[name]:.4f}")

    # Every required family must complete CV. Degenerate but valid families
    # remain recorded and are excluded below by the AUROC <= 0.5 guard.
    _require_complete_cv_evaluation(models, cv_auroc, cv_errors)

    # Select the family with the highest TRAINING cross-validated AUROC,
    # excluding any degenerate family whose CV AUROC <= 0.5 (no better than
    # chance, e.g. a tree that collapses to predicting a single class).
    best_model_name = _select_best_model(cv_auroc, cv_errors)

    # Fit ONLY the selected family on the full training set (cross_val_score
    # above cloned/refit internally, so models[best_model_name] is unfitted).
    # The final single-fit model: preprocessing fitted on the outer-training
    # rows, then the estimator. Bundled as a pipeline so the scoring artifact
    # carries its own preprocessor.
    best_estimator = _model_pipeline(preprocessor, models[best_model_name])
    best_estimator.fit(raw_train, y_train)
    fitted = {best_model_name: best_estimator}

    # Explicit five-fold isotonic fold ensembling on training data only.
    # Each fold fits one estimator/calibrator pair; their test probabilities
    # are averaged. No calibration decision uses the held-out test outcomes.
    #
    # When clusters exist, integer cv=5 is replaced by an explicit list of
    # five StratifiedGroupKFold splits so that one patient's encounters
    # cannot straddle a calibration fold. This preserves five-fold isotonic
    # fold ensembling exactly; only the fold ASSIGNMENT rule changes.
    if clusters is None:
        calibration_cv = 5
        calibration_protocol = CALIBRATION_PROTOCOL
    else:
        calibration_cv = list(
            StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=random_state
            ).split(raw_train, y_train, groups=clusters_train.to_numpy())
        )
        calibration_protocol = GROUPED_CALIBRATION_PROTOCOL
        _assert_no_cluster_leakage_in_folds(
            calibration_cv, clusters_train.to_numpy(), "calibration"
        )
    # Each of the five calibration folds clones the whole pipeline, so every
    # fold-specific calibrated model carries its OWN fitted preprocessor,
    # learned only from that fold's training portion.
    calibrated = _make_calibrated_classifier(best_estimator, cv=calibration_cv)
    calibrated.fit(raw_train, y_train)
    prob_calibrated = calibrated.predict_proba(raw_test)[:, 1]
    prob_uncalibrated = best_estimator.predict_proba(raw_test)[:, 1]

    # FINAL, one-time test-set evaluation of the selected model (reporting
    # only; this number was NOT used to choose the model).
    #
    # Note on calibration and ranking: each fold's isotonic map is monotone, but
    # the reported probability is the AVERAGE of five such maps, and an average
    # of monotone functions of the same score is monotone in that score only up
    # to the ties isotonic regression introduces. Calibrated and uncalibrated
    # AUROC can therefore differ slightly. Both are recorded rather than assumed
    # equal; the manuscript reports the calibrated model, which is the model
    # every other reported quantity is computed from.
    final_test_auroc = float(roc_auc_score(y_test_arr, prob_calibrated))
    uncalibrated_test_auroc = float(roc_auc_score(y_test_arr, prob_uncalibrated))
    # Backward-compatible key: downstream callers read
    # result["test_auroc"][result["best_model_name"]].
    test_auroc = {best_model_name: final_test_auroc}

    if verbose:
        print(f"\n  Selected family: '{best_model_name}' "
              f"(train CV AUROC={cv_auroc[best_model_name]:.4f})")
        print(f"  Final one-time test AUROC (calibrated) = {final_test_auroc:.4f}")
        print()

    # All downstream fairness analysis uses the CALIBRATED selected model.
    prob = prob_calibrated

    threshold_request = threshold
    if threshold == "prevalence":
        threshold = float(y_train.mean())
        threshold_protocol = PRIMARY_THRESHOLD_PROTOCOL
        if verbose:
            print(f"  threshold='prevalence' -> using training prevalence {threshold:.4f}\n")
    else:
        if not allow_nonprimary_threshold:
            raise ValueError(
                "Primary analyses must use threshold='prevalence'. Set "
                "allow_nonprimary_threshold=True only for a labelled "
                "robustness/exploratory analysis."
            )
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("A numeric threshold must be finite and in [0, 1]")
        threshold_protocol = "explicit_numeric_nonprimary"

    pred = (prob >= threshold).astype(int)

    # ------------------------------------------------------------------
    # 4. Raw per-subgroup metrics + fairness gaps
    # ------------------------------------------------------------------
    mask_a = g_test_arr == group_a_value
    mask_b = g_test_arr == group_b_value

    metrics_a = fm.subgroup_metrics(y_test_arr[mask_a], prob[mask_a], pred[mask_a])
    metrics_b = fm.subgroup_metrics(y_test_arr[mask_b], prob[mask_b], pred[mask_b])
    subgroup_metric_cis = {
        "A": fm.threshold_metric_confidence_intervals(metrics_a),
        "B": fm.threshold_metric_confidence_intervals(metrics_b),
    }
    gaps = fm.fairness_gaps(metrics_a, metrics_b)

    if verbose:
        print("=" * 70)
        print(f"STEP 3: Raw subgroup metrics  ({label_a} = A, {label_b} = B, gap = A - B)")
        print("=" * 70)
        rows = []
        for key in ["n", "n_pos", "prevalence", "auroc", "auprc",
                     "sensitivity", "specificity", "fpr", "fnr",
                     "ppv", "npv", "f1", "predicted_positive_rate",
                     "brier", "ece", "calib_intercept", "calib_slope"]:
            rows.append([key, metrics_a[key], metrics_b[key],
                         metrics_a[key] - metrics_b[key] if key not in ("n", "n_pos") else np.nan])
        raw_table = pd.DataFrame(rows, columns=["metric", label_a, label_b, "gap (A-B)"])
        print(raw_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print("Fairness-criterion gaps:")
        for k, v in gaps.items():
            print(f"  {k:28s} = {v:+.4f}")
        print()

    # ------------------------------------------------------------------
    # 5. Prevalence adjustment
    # ------------------------------------------------------------------
    prev_adj = adj.prevalence_adjusted_metrics(metrics_a, metrics_b)

    if verbose:
        print("=" * 70)
        print("STEP 4: Prevalence adjustment")
        print("=" * 70)
        print(f"Common target prevalence: {prev_adj['target_prevalence']:.4f}")
        print(f"(Sensitivity/specificity per group are held fixed; PPV, NPV, and")
        print(f" predicted-positive-rate are recomputed at the common prevalence.)\n")
        for key in ("ppv", "npv", "predicted_positive_rate"):
            d = prev_adj[key]
            print(f"  {key}:")
            print(f"    raw gap      (A-B) = {d['raw_gap']:+.4f}   "
                  f"({label_a}={d['raw_a']:.4f}, {label_b}={d['raw_b']:.4f})")
            print(f"    adjusted gap (A-B) = {d['adjusted_gap']:+.4f}   "
                  f"({label_a}={d['adjusted_a']:.4f}, {label_b}={d['adjusted_b']:.4f})")
            print(f"    attenuation        = {d['attenuation_pct']:.1f}%\n")
        di = prev_adj["disparate_impact_ratio"]
        print(f"  disparate impact ratio: raw={di['raw']:.4f}  adjusted={di['adjusted']:.4f}")
        print()

    # ------------------------------------------------------------------
    # 6. Bootstrap CIs for raw and prevalence-adjusted gaps
    # ------------------------------------------------------------------
    # All uncertainty for this comparison comes from ONE set of replicates.
    #
    # For a dataset with repeated encounters per patient the resampling unit is
    # the PATIENT, drawn once per replicate over the whole held-out set, with
    # the resulting multiplicity applied to every encounter of that patient in
    # whichever subgroup it falls. That keeps a patient recorded under more
    # than one demographic label from receiving two independent multiplicities,
    # which is what a subgroup-separate draw would do, and it makes the
    # subgroup estimates, their gap, discrimination, and calibration share one
    # coherent resampling distribution.
    #
    # Datasets with one row per respondent keep the established
    # subgroup-separate row-level design, which preserves each subgroup size.
    # ------------------------------------------------------------------
    cluster_test_arr = None if clusters is None else clusters_test.to_numpy()
    resampled = unc.bootstrap_comparison(
        y_test_arr, prob, pred, mask_a, mask_b,
        cluster_ids=cluster_test_arr,
        n_boot=n_boot, seed=random_state,
        target_prevalence=prev_adj["target_prevalence"],
    )
    bootstrap_protocol = resampled["design"]["resampling_design"]

    def _interval(block):
        return {
            "point": block["point"],
            "ci_low": block["ci_low"],
            "ci_high": block["ci_high"],
            "n_valid_replicates": block["n_valid_replicates"],
            "n_failed_replicates": block["n_failed_replicates"],
        }

    boot_results = {k: _interval(v) for k, v in resampled["gaps"].items()}
    boot_results_adj = {
        k: _interval(v) for k, v in resampled["prevalence_adjusted_gaps"].items()
    }

    # Subgroup-metric intervals. Where encounters are clustered, the binomial
    # (Wilson) interval is not valid because encounters are not independent
    # trials, so the cluster bootstrap replaces it. Independent-row datasets
    # keep Wilson.
    if clusters is None:
        subgroup_metric_cis = {
            "A": fm.threshold_metric_confidence_intervals(metrics_a),
            "B": fm.threshold_metric_confidence_intervals(metrics_b),
        }
        subgroup_ci_method = "wilson"
    else:
        def _cluster_cis(side, metrics):
            out = {}
            for key in ("prevalence", "sensitivity", "specificity", "ppv", "npv"):
                block = resampled["subgroup_" + side][key]
                numerator, denominator = _metric_counts(metrics, key)
                out[key] = {
                    "point": block["point"],
                    "ci_low": block["ci_low"],
                    "ci_high": block["ci_high"],
                    "method": "patient_cluster_bootstrap",
                    "numerator": numerator,
                    "denominator": denominator,
                }
            return out
        subgroup_metric_cis = {
            "A": _cluster_cis("a", metrics_a),
            "B": _cluster_cis("b", metrics_b),
        }
        subgroup_ci_method = "patient_cluster_bootstrap"

    auroc_ci = {
        "overall": _interval(resampled["discrimination"]["overall"]),
        "A": _interval(resampled["discrimination"]["a"]),
        "B": _interval(resampled["discrimination"]["b"]),
    }
    for block in auroc_ci.values():
        block.update({
            "method": resampled["design"]["ci_method"],
            "ci_level": resampled["design"]["ci_level"],
            "n_boot": resampled["design"]["n_boot"],
            "resampling_unit": resampled["design"]["resampling_unit"],
            "n_units": resampled["design"]["n_resampling_units"],
            "n_undefined_replicates": block["n_failed_replicates"],
        })
    auroc_gap_ci = _interval(resampled["discrimination"]["gap"])
    auroc_gap_ci.update({
        "method": resampled["design"]["ci_method"],
        "resampling_unit": resampled["design"]["resampling_unit"],
        "n_boot": resampled["design"]["n_boot"],
    })

    overall_metrics = fm.subgroup_metrics(y_test_arr, prob, pred)
    calibration_intervals = resampled["calibration"]

    if verbose:
        design = resampled["design"]
        print("=" * 70)
        print("STEP 5: Bootstrap 95% CIs "
              "(n_boot={}, unit={})".format(n_boot, design["resampling_unit"]))
        print("=" * 70)
        print("  design = {}, units = {:,}, units spanning both subgroups = {}"
              .format(design["resampling_design"],
                      design["n_resampling_units"],
                      design["n_units_spanning_subgroups"]))
        for key, res in boot_results.items():
            sig = "*" if res["ci_low"] > 0 or res["ci_high"] < 0 else " "
            print("  raw {:24s} gap = {:+.4f}  95% CI [{:+.4f}, {:+.4f}] {}"
                  .format(key, res["point"], res["ci_low"], res["ci_high"], sig))
        for key, res in boot_results_adj.items():
            sig = "*" if res["ci_low"] > 0 or res["ci_high"] < 0 else " "
            print("  prevalence-adjusted {:12s} gap = {:+.4f}  "
                  "95% CI [{:+.4f}, {:+.4f}] {}"
                  .format(key, res["point"], res["ci_low"], res["ci_high"], sig))
        print("  (* = 95% CI excludes 0)")
        print()
        for key, name in (("overall", "Overall"), ("A", label_a), ("B", label_b)):
            c = auroc_ci[key]
            print("  AUROC {:22s} = {:.4f}  95% CI [{:.4f}, {:.4f}]"
                  .format(name, c["point"], c["ci_low"], c["ci_high"]))
        for key, name in (("overall", "Overall"), ("a", label_a), ("b", label_b)):
            c = calibration_intervals[key]
            if np.isnan(c["slope"]["point"]):
                slope = "NA"
            else:
                slope = "{:.4f} [{:.4f}, {:.4f}]".format(
                    c["slope"]["point"], c["slope"]["ci_low"], c["slope"]["ci_high"])
            print("  Calibration {:20s} Brier={:.4f}  slope={}  [{}, failed={}]"
                  .format(name, c["brier"]["point"], slope, c["status"],
                          c["slope"]["n_failed_replicates"]))
        print()

    result = {
        "dataset_target_col": target_col,
        "group_col": group_col,
        "group_labels": {group_a_value: label_a, group_b_value: label_b},
        "models": fitted,
        "calibrated_model": calibrated,
        "best_model_name": best_model_name,
        "test_auroc": test_auroc,          # {selected_family: final one-time test AUROC}
        "cv_auroc": cv_auroc,              # training-CV AUROC for every family (selection basis)
        "cv_auroc_details": cv_auroc_details,
        "candidate_model_families": list(models),
        "final_test_auroc": final_test_auroc,
        "uncalibrated_test_auroc": uncalibrated_test_auroc,
        "overall_metrics": overall_metrics,
        "auroc_ci": auroc_ci,
        "auroc_gap_ci": auroc_gap_ci,
        "calibration_intervals": calibration_intervals,
        "resampling_design": resampled["design"],
        "subgroup_ci_method": subgroup_ci_method,
        "calibration_protocol": calibration_protocol,
        "selection_cv_protocol": selection_cv_protocol,
        "split_protocol": split_protocol,
        "bootstrap_protocol": bootstrap_protocol,
        "split_diagnostics": split_diagnostics,
        "preprocessing_schema": preprocessor.schema(),
        "feature_names": design_feature_names,
        "preprocessor": preprocessor,
        "X_test": X_test,
        "X_train": X_train,
        "raw_test": raw_test,
        "raw_train": raw_train,
        "threshold_protocol": threshold_protocol,
        "threshold_request": threshold_request,
        "case_mix_inference_protocol": CASE_MIX_INFERENCE_PROTOCOL,
        "runtime_provenance": runtime_provenance(),
        "n_test": len(X_test),
        "train_index": X_train.index.to_numpy(),
        "test_index": X_test.index.to_numpy(),
        "decision_threshold": threshold,
        "y_test": y_test_arr,
        "g_test": g_test_arr,
        "prob": prob,
        "prob_uncalibrated": prob_uncalibrated,
        "pred": pred,
        "subgroup_metrics": {"A": metrics_a, "B": metrics_b},
        "subgroup_metric_cis": subgroup_metric_cis,
        "fairness_gaps": gaps,
        "prevalence_adjustment": prev_adj,
        "bootstrap_raw": boot_results,
        "bootstrap_prevalence_adjusted": boot_results_adj,
    }

    # ------------------------------------------------------------------
    # 7. Exploratory calibration/threshold analysis (optional)
    # ------------------------------------------------------------------
    # IMPORTANT: the target sensitivity and subgroup thresholds below are
    # derived from the final test outcomes and then described on those same
    # observations. This is a post-hoc operating-point illustration, not an
    # independently evaluated or deployment-ready threshold policy.
    # ------------------------------------------------------------------
    if run_calibration_adjustment:
        if verbose:
            print("=" * 70)
            print("STEP 6: Exploratory calibration/threshold analysis")
            print("=" * 70)
            print("  DESCRIPTIVE ONLY: thresholds are derived and evaluated on the")
            print("  same final test outcomes; this is not an independently validated policy.\n")

        pooled_sensitivity = fm.threshold_metrics(y_test_arr, pred)["sensitivity"]
        groups_present = [group_a_value, group_b_value]
        eq_sens_thresholds = adj.equal_sensitivity_thresholds(
            y_test_arr, prob, g_test_arr, groups_present, pooled_sensitivity
        )

        pred_eq_sens = np.zeros_like(pred)
        for val in groups_present:
            mask = g_test_arr == val
            pred_eq_sens[mask] = (prob[mask] >= eq_sens_thresholds[val]).astype(int)

        metrics_a_eq = fm.subgroup_metrics(y_test_arr[mask_a], prob[mask_a], pred_eq_sens[mask_a])
        metrics_b_eq = fm.subgroup_metrics(y_test_arr[mask_b], prob[mask_b], pred_eq_sens[mask_b])
        gaps_eq = fm.fairness_gaps(metrics_a_eq, metrics_b_eq)

        calib_adjustment = {
            "analysis_status": "exploratory_descriptive",
            "threshold_derivation_data": "final_test_set",
            "independent_evaluation": False,
            "warning": (
                "Thresholds and target sensitivity were derived and evaluated "
                "on the same final test outcomes."
            ),
            "shared_threshold": threshold,
            "target_sensitivity": pooled_sensitivity,
            "equal_sensitivity_thresholds": {
                label_a: eq_sens_thresholds[group_a_value],
                label_b: eq_sens_thresholds[group_b_value],
            },
            "metrics_shared_threshold": gaps,
            "metrics_equal_sensitivity": gaps_eq,
        }
        result["calibration_adjustment"] = calib_adjustment

        if verbose:
            print(f"Shared threshold = {threshold}")
            print(f"Equal-sensitivity thresholds (target sens={pooled_sensitivity:.3f}): "
                  f"{label_a}={eq_sens_thresholds[group_a_value]:.4f}, "
                  f"{label_b}={eq_sens_thresholds[group_b_value]:.4f}\n")
            print(f"{'gap':28s} {'shared thresh':>15s} {'equal-sens thresh':>20s}")
            for k in gaps:
                print(f"{k:28s} {gaps[k]:>15.4f} {gaps_eq[k]:>20.4f}")
            print()

    # ------------------------------------------------------------------
    # 8. Case-mix waterfall (optional)
    # ------------------------------------------------------------------
    if run_case_mix and covariate_blocks:
        if verbose:
            print("=" * 70)
            print("STEP 7: Case-mix waterfall (outcome gap, not model gap)")
            print("=" * 70)

        # The waterfall is a descriptive outcome ~ group + covariates
        # regression over all compared rows; it is not a train/test exercise
        # and its estimand is unchanged. It does, however, need a design
        # matrix, so it uses the SAME training-derived preprocessing applied
        # to all compared rows, and block definitions naming a categorical
        # source (e.g. "race") expand to that source's learned indicators.
        case_mix_design = preprocessor.transform(work, partition="case_mix_all_rows")
        case_mix_frame = pd.concat(
            [case_mix_design, work[[target_col, group_col]]], axis=1
        )
        expanded_blocks = {
            block: preprocessor.expand_columns(columns)
            for block, columns in covariate_blocks.items()
        }
        waterfall = adj.case_mix_waterfall(
            case_mix_frame, target_col, group_col, group_a_value, group_b_value,
            expanded_blocks, random_state=random_state
        )
        result["case_mix"] = waterfall

        if verbose:
            print(waterfall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print()

    _persist_pipeline_provenance(
        result,
        target_col=target_col,
        group_col=group_col,
        group_a_value=group_a_value,
        group_b_value=group_b_value,
        label_a=label_a,
        label_b=label_b,
        random_state=random_state,
        n_rows=len(work),
        n_features=len(design_feature_names),
        analysis_label=analysis_label,
    )

    return result
