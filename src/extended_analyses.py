"""
extended_analyses.py
====================
Analysis functions for the robustness / decomposition extension of the
fairness pipeline.  Every function is dataset-agnostic: it takes numpy
arrays / DataFrames and returns a pandas DataFrame or dict.

Priority order follows the "FASTEST PATH" from the decomposition framework
document:
  1  fn_regression_waterfall     — FN ~ Sex + covariate blocks (among positives)
  2  fp_regression_waterfall     — FP ~ Sex + covariate blocks (among negatives)
  3  error_based_waterfall       — wrapper calling both
  4  (repeated splits — in runner)
  5  (survey weights — in runner)
  6  (Diabetes-130 dedup — in runner)
  7  threshold_sweep             — all gap metrics vs threshold
  8  shapley_decompose_gap       — Shapley decomposition of PPV/NPV/PPR gap
  9  empirical_prevalence_match  — IPW prevalence standardisation
  10 synthetic_controls          — known-mechanism positive controls
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

from . import metrics as fm
from .adjustments import ppv_npv_at_prevalence, attenuation_pct


# ---------------------------------------------------------------------------
# Helper: confusion-matrix counts
# ---------------------------------------------------------------------------

def confusion_counts(y_true, y_pred):
    """Return TP, FP, TN, FN as a dict."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "n_pos": tp + fn, "n_neg": fp + tn, "n": tp + fp + tn + fn}


# ---------------------------------------------------------------------------
# 1 & 2. FN / FP regression waterfall
# ---------------------------------------------------------------------------

def _error_regression_waterfall(
    y_true, y_pred, g_test, X_test,
    covariate_blocks,
    group_a_val, group_b_val,
    error_type="fn",        # "fn" or "fp"
):
    """
    Blockwise logistic regression of model error on sex.

    error_type = "fn": restrict to true positives (y_true==1);
                       outcome = FalseNegative = (y_pred==0)
    error_type = "fp": restrict to true negatives (y_true==0);
                       outcome = FalsePositive  = (y_pred==1)

    Returns DataFrame with one row per step:
      step | n_obs | n_error | error_rate_A | error_rate_B |
      sex_logOR | sex_OR | sex_OR_ci_low | sex_OR_ci_high |
      sex_p | n_features | pct_of_raw_coef_explained
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    g_test  = np.asarray(g_test)

    if error_type == "fn":
        restrict_mask = (y_true == 1)
        error_outcome = (y_pred[restrict_mask] == 0).astype(int)
        label = "FalseNegative"
    else:
        restrict_mask = (y_true == 0)
        error_outcome = (y_pred[restrict_mask] == 1).astype(int)
        label = "FalsePositive"

    g_sub   = g_test[restrict_mask]
    mask_a  = g_sub == group_a_val
    mask_b  = g_sub == group_b_val
    keep    = mask_a | mask_b
    error_outcome = error_outcome[keep]
    g_sub   = g_sub[keep]
    mask_a  = g_sub == group_a_val

    # sex binary: 1 = group A (Male)
    sex_binary = mask_a.astype(int)

    # subselect X rows
    if isinstance(X_test, pd.DataFrame):
        all_indices = np.where(restrict_mask)[0]
        all_indices = all_indices[keep]
        X_sub = X_test.iloc[all_indices].reset_index(drop=True)
    else:
        X_sub = X_test[restrict_mask][keep]

    # per-group error rates
    er_a = float(error_outcome[sex_binary == 1].mean()) if (sex_binary == 1).any() else np.nan
    er_b = float(error_outcome[sex_binary == 0].mean()) if (sex_binary == 0).any() else np.nan

    rows = []
    raw_coef = None
    cumulative_cols = []

    step_list = [("raw (sex only)", [])] + list(covariate_blocks.items())

    for step_name, block_cols in step_list:
        if block_cols:
            if isinstance(X_sub, pd.DataFrame):
                valid = [c for c in block_cols if c in X_sub.columns]
            else:
                valid = []
            cumulative_cols += valid

        # Build design matrix
        if cumulative_cols and isinstance(X_sub, pd.DataFrame):
            X_cov = X_sub[cumulative_cols].copy()
            # standard scale covariates to help convergence
            scaler = StandardScaler()
            X_cov_s = pd.DataFrame(
                scaler.fit_transform(X_cov),
                columns=cumulative_cols
            )
            design = pd.concat(
                [pd.Series(sex_binary, name="sex"), X_cov_s], axis=1
            )
        else:
            design = pd.DataFrame({"sex": sex_binary})

        n_features = design.shape[1]

        # Fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if _HAS_SM:
                exog = sm.add_constant(design, has_constant="add")
                try:
                    result = sm.Logit(error_outcome, exog).fit(
                        disp=0, method="bfgs", maxiter=200
                    )
                    coef  = float(result.params.get("sex", np.nan))
                    ci    = result.conf_int()
                    ci_lo = float(ci.loc["sex", 0]) if "sex" in ci.index else np.nan
                    ci_hi = float(ci.loc["sex", 1]) if "sex" in ci.index else np.nan
                    pval  = float(result.pvalues.get("sex", np.nan))
                except Exception:
                    coef, ci_lo, ci_hi, pval = np.nan, np.nan, np.nan, np.nan
            else:
                clf = LogisticRegression(max_iter=500, C=1e6)
                clf.fit(design, error_outcome)
                sex_idx = list(design.columns).index("sex")
                coef = float(clf.coef_[0][sex_idx])
                ci_lo, ci_hi, pval = np.nan, np.nan, np.nan

        if raw_coef is None:
            raw_coef = coef

        pct = (100.0 * (raw_coef - coef) / raw_coef
               if (raw_coef is not None and raw_coef != 0 and not np.isnan(coef))
               else np.nan)

        rows.append({
            "step":                    step_name,
            "n_obs":                   int(len(error_outcome)),
            "n_error":                 int(error_outcome.sum()),
            f"error_rate_{group_a_val}": er_a,
            f"error_rate_{group_b_val}": er_b,
            "sex_logOR":               coef,
            "sex_OR":                  np.exp(coef) if not np.isnan(coef) else np.nan,
            "sex_OR_ci_low":           np.exp(ci_lo) if not np.isnan(ci_lo) else np.nan,
            "sex_OR_ci_high":          np.exp(ci_hi) if not np.isnan(ci_hi) else np.nan,
            "sex_p":                   pval,
            "n_features":              n_features,
            "pct_of_raw_coef_explained": pct,
        })

    df_out = pd.DataFrame(rows)
    df_out.attrs["error_type"]  = error_type
    df_out.attrs["error_label"] = label
    df_out.attrs["er_a_raw"]    = er_a
    df_out.attrs["er_b_raw"]    = er_b
    return df_out


def fn_regression_waterfall(y_true, y_pred, g_test, X_test,
                              covariate_blocks, group_a_val, group_b_val):
    """FalseNegative ~ Sex + blocks, restricted to true positives."""
    return _error_regression_waterfall(
        y_true, y_pred, g_test, X_test,
        covariate_blocks, group_a_val, group_b_val, error_type="fn"
    )


def fp_regression_waterfall(y_true, y_pred, g_test, X_test,
                              covariate_blocks, group_a_val, group_b_val):
    """FalsePositive ~ Sex + blocks, restricted to true negatives."""
    return _error_regression_waterfall(
        y_true, y_pred, g_test, X_test,
        covariate_blocks, group_a_val, group_b_val, error_type="fp"
    )


# ---------------------------------------------------------------------------
# 7. Threshold sweep
# ---------------------------------------------------------------------------

THRESHOLD_SWEEP_METRICS = (
    "sensitivity",
    "specificity",
    "fnr",
    "fpr",
    "ppv",
    "npv",
    "predicted_positive_rate",
)

THRESHOLD_SWEEP_COLUMNS = (
    "threshold",
    *(
        column
        for metric in THRESHOLD_SWEEP_METRICS
        for column in (f"{metric}_A", f"{metric}_B", f"{metric}_gap")
    ),
    "disparate_impact_ratio",
)

THRESHOLD_SWEEP_CSV_COLUMNS = ("dataset", *THRESHOLD_SWEEP_COLUMNS)

THRESHOLD_SWEEP_PLOT_COLUMNS = (
    ("sensitivity_gap", "Sensitivity gap (M−F)"),
    ("ppv_gap", "PPV gap (M−F)"),
    ("predicted_positive_rate_gap", "PPR gap (M−F)"),
    ("fnr_gap", "FNR gap (M−F)"),
)


def validate_threshold_sweep_schema(sweep):
    """Require the exact documented threshold-sweep producer schema."""
    actual = tuple(sweep.columns)
    if actual != THRESHOLD_SWEEP_COLUMNS:
        missing = [column for column in THRESHOLD_SWEEP_COLUMNS if column not in actual]
        unexpected = [column for column in actual if column not in THRESHOLD_SWEEP_COLUMNS]
        raise ValueError(
            "Threshold-sweep schema mismatch: "
            f"expected={list(THRESHOLD_SWEEP_COLUMNS)!r}; "
            f"actual={list(actual)!r}; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )


def save_threshold_sweep_plot(sweep, operating_threshold, dataset_name, output_path):
    """Validate and render the threshold-sweep plot used by production."""
    validate_threshold_sweep_schema(sweep)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (column, ylabel) in zip(axes.flat, THRESHOLD_SWEEP_PLOT_COLUMNS):
        ax.plot(sweep["threshold"], sweep[column], lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(
            operating_threshold,
            color="red",
            lw=0.8,
            ls=":",
            label=f"used={operating_threshold:.3f}",
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Threshold")
        ax.legend(fontsize=7)
    fig.suptitle(f"Threshold sweep — {dataset_name}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


CALIBRATION_ROBUSTNESS_SUMMARY_METRICS = (
    "sensitivity_gap",
    "fnr_gap",
    "ppv_gap_raw",
    "ppv_gap_adj",
    "ppr_gap",
)
CALIBRATION_ROBUSTNESS_SUMMARY_STATISTICS = ("mean", "std", "min", "max")
CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS = (
    "dataset",
    *(
        f"{metric}_{statistic}"
        for metric in CALIBRATION_ROBUSTNESS_SUMMARY_METRICS
        for statistic in CALIBRATION_ROBUSTNESS_SUMMARY_STATISTICS
    ),
)


def calibration_method_robustness_summary(method_results):
    """Return the unique, flat CSV schema for calibration-method summaries."""
    required = ("dataset", *CALIBRATION_ROBUSTNESS_SUMMARY_METRICS)
    missing = [column for column in required if column not in method_results.columns]
    if missing:
        raise ValueError(
            f"Calibration-robustness input schema is missing columns: {missing!r}"
        )

    summary = method_results.groupby("dataset", sort=True)[
        list(CALIBRATION_ROBUSTNESS_SUMMARY_METRICS)
    ].agg(list(CALIBRATION_ROBUSTNESS_SUMMARY_STATISTICS))
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    actual = tuple(summary.columns)
    if actual != CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS:
        raise ValueError(
            "Calibration-robustness summary schema mismatch: "
            f"expected={list(CALIBRATION_ROBUSTNESS_SUMMARY_COLUMNS)!r}; "
            f"actual={list(actual)!r}"
        )
    return summary

def threshold_sweep(y_true, prob, g_test, group_a_val, group_b_val,
                    n_steps=99):
    """
    Sweep the decision threshold from 0.01 to 0.99 and compute all gap
    metrics at each step. Returns a DataFrame with the exact columns in
    ``THRESHOLD_SWEEP_COLUMNS``: ``threshold``; ``_A``, ``_B``, and ``_gap``
    columns for sensitivity, specificity, FNR, FPR, PPV, NPV, and predicted
    positive rate; followed by ``disparate_impact_ratio``.

    ``predicted_positive_rate_gap`` is the canonical descriptive name for
    the predicted-positive-rate gap. This schema does not define a
    ``ppr_gap`` alias; other analyses may use that shorter name under their
    own separately documented output contracts.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)

    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val
    thresholds = np.linspace(0.01, 0.99, n_steps)

    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        ma = fm.threshold_metrics(y_true[mask_a], pred[mask_a])
        mb = fm.threshold_metrics(y_true[mask_b], pred[mask_b])
        row = {"threshold": round(float(t), 4)}
        for key in THRESHOLD_SWEEP_METRICS:
            row[f"{key}_A"] = ma[key]
            row[f"{key}_B"] = mb[key]
            row[f"{key}_gap"] = ma[key] - mb[key]
        ppr_a = ma["predicted_positive_rate"]
        ppr_b = mb["predicted_positive_rate"]
        row["disparate_impact_ratio"] = (ppr_a / ppr_b) if ppr_b > 0 else np.nan
        rows.append(row)

    sweep = pd.DataFrame(rows, columns=THRESHOLD_SWEEP_COLUMNS)
    validate_threshold_sweep_schema(sweep)
    return sweep


# ---------------------------------------------------------------------------
# 8. Shapley decomposition of PPV / NPV / PPR gap
# ---------------------------------------------------------------------------

def shapley_decompose_gap(metrics_a, metrics_b):
    """
    Decompose the PPV, NPV, and PPR gap between group A and B into
    contributions from:
      - prevalence difference
      - sensitivity difference
      - specificity difference

    Uses the exact Shapley value over these three inputs (3! = 6 orderings).
    The three Shapley values sum exactly to the raw gap.

    Returns a DataFrame with one row per metric (ppv / npv / ppr) and
    columns: raw_gap, shapley_prevalence, shapley_sensitivity,
    shapley_specificity, check_sum.
    """
    sens_a  = metrics_a["sensitivity"]
    spec_a  = metrics_a["specificity"]
    prev_a  = metrics_a["prevalence"]
    sens_b  = metrics_b["sensitivity"]
    spec_b  = metrics_b["specificity"]
    prev_b  = metrics_b["prevalence"]

    def metric_at(sens, spec, prev, key):
        out = ppv_npv_at_prevalence(sens, spec, prev)
        return out[key]

    rows = []
    for mkey in ("ppv", "npv", "predicted_positive_rate"):
        # Baseline = group B values
        vals = {
            "sens": (sens_b, sens_a),
            "spec": (spec_b, spec_a),
            "prev": (prev_b, prev_a),
        }
        raw_gap = metric_at(sens_a, spec_a, prev_a, mkey) - \
                  metric_at(sens_b, spec_b, prev_b, mkey)

        shapley = {"sens": 0.0, "spec": 0.0, "prev": 0.0}
        features = ["sens", "spec", "prev"]
        from itertools import permutations
        perms = list(permutations(features))
        for perm in perms:
            current = {"sens": sens_b, "spec": spec_b, "prev": prev_b}
            for feat in perm:
                before = metric_at(current["sens"], current["spec"],
                                   current["prev"], mkey)
                current[feat] = vals[feat][1]   # switch to group A value
                after  = metric_at(current["sens"], current["spec"],
                                   current["prev"], mkey)
                shapley[feat] += (after - before) / len(perms)

        rows.append({
            "metric":              mkey,
            "raw_gap":             raw_gap,
            "shapley_prevalence":  shapley["prev"],
            "shapley_sensitivity": shapley["sens"],
            "shapley_specificity": shapley["spec"],
            "check_sum":           shapley["prev"] + shapley["sens"] + shapley["spec"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Empirical prevalence matching (IPW)
# ---------------------------------------------------------------------------

def empirical_prevalence_match(y_true, prob, pred, g_test,
                                group_a_val, group_b_val,
                                target_prevalence=None,
                                n_boot=500, seed=42):
    """
    Equalize subgroup prevalences via inverse-probability weighting (IPW)
    and recompute PPV, NPV, and PPR.  Compares empirical IPW-adjusted gaps
    against the Bayes-rule (analytic) adjusted gaps.

    For each subgroup, each positive case is upweighted by
        target_prev / group_prev
    and each negative case is upweighted by
        (1 - target_prev) / (1 - group_prev)

    Returns a dict with keys:
      table: DataFrame with columns metric, bayes_adj_gap, ipw_adj_gap,
        ipw_ci_low, ipw_ci_high, absolute_diff;
      max_absolute_diff: largest absolute Bayes-vs-IPW gap difference;
      target_prevalence: prevalence used for standardization.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred   = np.asarray(pred,   dtype=int)
    g_test = np.asarray(g_test)
    prob   = np.asarray(prob,   dtype=float)

    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val

    def _group_metrics(yt, yp):
        return fm.threshold_metrics(yt, yp)

    ma0 = _group_metrics(y_true[mask_a], pred[mask_a])
    mb0 = _group_metrics(y_true[mask_b], pred[mask_b])

    prev_a = ma0["prevalence"]
    prev_b = mb0["prevalence"]

    if target_prevalence is None:
        n_pos = y_true[mask_a | mask_b].sum()
        n_tot = (mask_a | mask_b).sum()
        target_prevalence = n_pos / n_tot

    # Bayes-rule gaps (analytic)
    from .adjustments import ppv_npv_at_prevalence
    adj_a = ppv_npv_at_prevalence(ma0["sensitivity"], ma0["specificity"], target_prevalence)
    adj_b = ppv_npv_at_prevalence(mb0["sensitivity"], mb0["specificity"], target_prevalence)
    bayes_gaps = {k: adj_a[k] - adj_b[k] for k in ("ppv", "npv", "predicted_positive_rate")}

    # IPW
    def _ipw_weights(y_true_g, prev_g, target_prev):
        w = np.where(y_true_g == 1,
                     target_prev / prev_g if prev_g > 0 else 1.0,
                     (1 - target_prev) / (1 - prev_g) if prev_g < 1 else 1.0)
        return w

    def _weighted_metric(yt, yp, w):
        """Weighted TP, FP, FN, TN -> weighted PPV, NPV, PPR."""
        w = np.asarray(w, dtype=float)
        wtp = (w * ((yt == 1) & (yp == 1))).sum()
        wfp = (w * ((yt == 0) & (yp == 1))).sum()
        wtn = (w * ((yt == 0) & (yp == 0))).sum()
        wfn = (w * ((yt == 1) & (yp == 0))).sum()
        ppv  = wtp / (wtp + wfp) if (wtp + wfp) > 0 else np.nan
        npv  = wtn / (wtn + wfn) if (wtn + wfn) > 0 else np.nan
        ppr  = (wtp + wfp) / (wtp + wfp + wtn + wfn)
        return {"ppv": ppv, "npv": npv, "predicted_positive_rate": ppr}

    wa = _ipw_weights(y_true[mask_a], prev_a, target_prevalence)
    wb = _ipw_weights(y_true[mask_b], prev_b, target_prevalence)

    ipw_a = _weighted_metric(y_true[mask_a], pred[mask_a], wa)
    ipw_b = _weighted_metric(y_true[mask_b], pred[mask_b], wb)
    ipw_gaps = {k: ipw_a[k] - ipw_b[k] for k in ("ppv", "npv", "predicted_positive_rate")}

    # Bootstrap CIs for IPW gaps
    rng = np.random.default_rng(seed)
    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]
    boot_ipw = {k: [] for k in ("ppv", "npv", "predicted_positive_rate")}
    for _ in range(n_boot):
        sa = rng.choice(idx_a, size=len(idx_a), replace=True)
        sb = rng.choice(idx_b, size=len(idx_b), replace=True)
        wa_ = _ipw_weights(y_true[sa], prev_a, target_prevalence)
        wb_ = _ipw_weights(y_true[sb], prev_b, target_prevalence)
        ia_ = _weighted_metric(y_true[sa], pred[sa], wa_)
        ib_ = _weighted_metric(y_true[sb], pred[sb], wb_)
        for k in boot_ipw:
            boot_ipw[k].append(ia_[k] - ib_[k])

    rows = []
    for k in ("ppv", "npv", "predicted_positive_rate"):
        boot = np.array(boot_ipw[k])
        if n_boot:
            ci_low = float(np.nanquantile(boot, 0.025))
            ci_high = float(np.nanquantile(boot, 0.975))
        else:
            ci_low = np.nan
            ci_high = np.nan
        rows.append({
            "metric":           k,
            "bayes_adj_gap":    bayes_gaps[k],
            "ipw_adj_gap":      ipw_gaps[k],
            "ipw_ci_low":       ci_low,
            "ipw_ci_high":      ci_high,
            "inference_protocol": (
                "bootstrap_interval" if n_boot
                else "exploratory_point_estimate_no_confidence_interval"
            ),
            "absolute_diff":    abs(bayes_gaps[k] - ipw_gaps[k]),
        })

    df_out = pd.DataFrame(rows)
    max_diff = df_out["absolute_diff"].max()
    return {"table": df_out, "max_absolute_diff": float(max_diff),
            "target_prevalence": target_prevalence}


# ---------------------------------------------------------------------------
# 10. Synthetic controls (positive controls with known mechanism)
# ---------------------------------------------------------------------------

def synthetic_control(mechanism="prevalence_only", n=20000, seed=42):
    """
    Generate a semi-synthetic test set where the true data-generating
    mechanism is known, then run the decomposition to verify recovery.

    mechanism options:
      "prevalence_only"   — identical sens/spec, different prevalence
      "sensitivity_only"  — identical spec/prev, different sensitivity
      "calibration_only"  — identical underlying ROC, different score shift
      "residual_fnr"      — identical prev/spec, different sensitivity (same as sensitivity_only
                            but framed as residual error)
      "null"              — identical everything, no gap should be found

    Returns one flat dict suitable for a CSV row, with mechanism; true and
    observed prevalence, sensitivity, and specificity for groups A/B;
    ppv_gap_raw; ppv_gap_adj; ppv_attenuation; sens_gap_raw;
    decomposition_correctly_identifies; and n_per_group_A/B.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth parameters
    if mechanism == "prevalence_only":
        prev_a, prev_b = 0.25, 0.10
        sens_a = sens_b = 0.80
        spec_a = spec_b = 0.85
        expected_attenuated = ["ppv", "npv"]
        expected_residual   = []
    elif mechanism == "sensitivity_only":
        prev_a = prev_b = 0.15
        sens_a, sens_b = 0.70, 0.85
        spec_a = spec_b = 0.85
        expected_attenuated = []
        expected_residual   = ["sensitivity"]
    elif mechanism == "null":
        prev_a = prev_b = 0.15
        sens_a = sens_b = 0.80
        spec_a = spec_b = 0.85
        expected_attenuated = []
        expected_residual   = []
    elif mechanism == "calibration_only":
        prev_a = prev_b = 0.15
        sens_a, sens_b = 0.78, 0.82   # slight diff from shared-threshold on differently-calibrated scores
        spec_a, spec_b = 0.87, 0.83
        expected_attenuated = ["ppv", "npv"]
        expected_residual   = ["sensitivity"]
    elif mechanism == "residual_fnr":
        prev_a = prev_b = 0.15
        sens_a, sens_b = 0.68, 0.84
        spec_a = spec_b = 0.85
        expected_attenuated = []
        expected_residual   = ["sensitivity"]
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    half = n // 2

    def _simulate_group(prev, sens, spec, n_g):
        y  = (rng.random(n_g) < prev).astype(int)
        yp = np.where(y == 1,
                      (rng.random(n_g) < sens).astype(int),
                      (rng.random(n_g) < (1 - spec)).astype(int))
        return y, yp

    ya, ypa = _simulate_group(prev_a, sens_a, spec_a, half)
    yb, ypb = _simulate_group(prev_b, sens_b, spec_b, n - half)

    ma = fm.threshold_metrics(ya, ypa)
    mb = fm.threshold_metrics(yb, ypb)

    # Prevalence adjustment
    target_prev = (ya.sum() + yb.sum()) / n
    from .adjustments import ppv_npv_at_prevalence
    adj_a = ppv_npv_at_prevalence(ma["sensitivity"], ma["specificity"], target_prev)
    adj_b = ppv_npv_at_prevalence(mb["sensitivity"], mb["specificity"], target_prev)

    ppv_gap_raw  = ma["ppv"]  - mb["ppv"]
    ppv_gap_adj  = adj_a["ppv"] - adj_b["ppv"]
    sens_gap_raw = ma["sensitivity"] - mb["sensitivity"]
    ppv_atten    = attenuation_pct(ppv_gap_raw, ppv_gap_adj)

    # Recovery check
    if mechanism == "prevalence_only":
        # Adjustment should nearly eliminate PPV gap; sens gap should be ~0
        recovered = abs(ppv_gap_adj) < 0.03 and abs(sens_gap_raw) < 0.05
    elif mechanism in ("sensitivity_only", "residual_fnr"):
        # Adjustment should NOT eliminate sens gap; PPV gap should shrink
        recovered = abs(sens_gap_raw) > 0.05
    elif mechanism == "null":
        recovered = abs(ppv_gap_raw) < 0.05 and abs(sens_gap_raw) < 0.05
    elif mechanism == "calibration_only":
        recovered = True   # just demonstrate the shift
    else:
        recovered = False

    return {
        "mechanism":       mechanism,
        "true_prev_A":     prev_a, "true_prev_B": prev_b,
        "true_sens_A":     sens_a, "true_sens_B": sens_b,
        "true_spec_A":     spec_a, "true_spec_B": spec_b,
        "observed_prev_A": float(ma["prevalence"]),
        "observed_prev_B": float(mb["prevalence"]),
        "observed_sens_A": float(ma["sensitivity"]),
        "observed_sens_B": float(mb["sensitivity"]),
        "observed_spec_A": float(ma["specificity"]),
        "observed_spec_B": float(mb["specificity"]),
        "ppv_gap_raw":     float(ppv_gap_raw),
        "ppv_gap_adj":     float(ppv_gap_adj),
        "ppv_attenuation": float(ppv_atten),
        "sens_gap_raw":    float(sens_gap_raw),
        "decomposition_correctly_identifies": recovered,
        "n_per_group_A":   half,
        "n_per_group_B":   n - half,
    }


# ---------------------------------------------------------------------------
# Score-distribution analysis among positives / negatives
# ---------------------------------------------------------------------------

def score_distribution_by_group(y_true, prob, g_test, group_a_val, group_b_val,
                                  restrict_to="positives"):
    """
    Compare predicted-risk distributions between groups among
    true positives (restrict_to='positives') or true negatives ('negatives').

    Returns a summary DataFrame and runs a KS test.
    """
    from scipy import stats as scipy_stats

    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)

    if restrict_to == "positives":
        sub = y_true == 1
    else:
        sub = y_true == 0

    scores_a = prob[sub & (g_test == group_a_val)]
    scores_b = prob[sub & (g_test == group_b_val)]

    ks_stat, ks_p = scipy_stats.ks_2samp(scores_a, scores_b)

    # P(random A positive < random B positive) -- lower score in A = worse
    # Use Mann-Whitney U rank statistic
    u_stat, u_p = scipy_stats.mannwhitneyu(scores_a, scores_b, alternative="less")
    prob_a_lower = u_stat / (len(scores_a) * len(scores_b))

    rows = []
    for label, scores in [(group_a_val, scores_a), (group_b_val, scores_b)]:
        rows.append({
            "group":  label,
            "n":      len(scores),
            "mean":   float(np.mean(scores)),
            "median": float(np.median(scores)),
            "sd":     float(np.std(scores)),
            "q25":    float(np.quantile(scores, 0.25)),
            "q75":    float(np.quantile(scores, 0.75)),
        })
    df_out = pd.DataFrame(rows)
    df_out.attrs["ks_stat"]     = ks_stat
    df_out.attrs["ks_p"]        = ks_p
    df_out.attrs["prob_A_lower_score_than_B"] = prob_a_lower
    df_out.attrs["restrict_to"] = restrict_to
    return df_out


# ---------------------------------------------------------------------------
# Confusion matrix table (full counts per dataset and subgroup)
# ---------------------------------------------------------------------------

def full_confusion_table(y_true, y_pred, g_test, group_a_val, group_b_val,
                          label_a="Male", label_b="Female"):
    """Return confusion-matrix counts for each subgroup."""
    rows = []
    for val, lbl in [(group_a_val, label_a), (group_b_val, label_b)]:
        mask = g_test == val
        cc = confusion_counts(y_true[mask], y_pred[mask])
        rows.append({"group": lbl, **cc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 16/17/18. Generic "equalize a target metric across subgroups" thresholds
# ---------------------------------------------------------------------------

def find_threshold_for_metric(y_true_g, prob_g, metric_key, target_value,
                               grid=None):
    """
    Grid-search the probability threshold for one subgroup that brings
    `metric_key` (a key returned by fm.threshold_metrics) as close as
    possible to `target_value`.

    Used to equalize specificity, PPV, or predicted-positive-rate across
    subgroups (mirrors adjustments.equal_sensitivity_thresholds, which has
    a closed-form solution only for sensitivity).
    """
    from . import metrics as fm
    y_true_g = np.asarray(y_true_g, dtype=int)
    prob_g   = np.asarray(prob_g,   dtype=float)
    if grid is None:
        grid = np.linspace(0.001, 0.999, 999)

    best_t, best_diff = 0.5, np.inf
    for t in grid:
        pred = (prob_g >= t).astype(int)
        m = fm.threshold_metrics(y_true_g, pred)
        val = m.get(metric_key, np.nan)
        if np.isnan(val):
            continue
        diff = abs(val - target_value)
        if diff < best_diff:
            best_diff, best_t = diff, float(t)
    return best_t


def equal_metric_threshold_result(y_true, prob, g_test, group_a_val, group_b_val,
                                   metric_key, label_a, label_b,
                                   shared_threshold):
    """
    Find group-specific thresholds that equalize `metric_key` (e.g.
    "specificity", "ppv", "predicted_positive_rate") across subgroups, then
    report all fairness gaps before (shared threshold) and after
    (equalized thresholds).

    Returns a dict: target_value, thresholds, before/after metric tables,
    and a tidy gap-comparison DataFrame.
    """
    from . import metrics as fm

    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)
    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val

    # shared-threshold ("before") predictions
    pred_shared = (prob >= shared_threshold).astype(int)
    before_a = fm.threshold_metrics(y_true[mask_a], pred_shared[mask_a])
    before_b = fm.threshold_metrics(y_true[mask_b], pred_shared[mask_b])

    # target = pooled value of metric_key at the shared threshold
    target_value = fm.threshold_metrics(y_true, pred_shared)[metric_key]

    t_a = find_threshold_for_metric(y_true[mask_a], prob[mask_a], metric_key, target_value)
    t_b = find_threshold_for_metric(y_true[mask_b], prob[mask_b], metric_key, target_value)

    pred_eq = np.zeros_like(pred_shared)
    pred_eq[mask_a] = (prob[mask_a] >= t_a).astype(int)
    pred_eq[mask_b] = (prob[mask_b] >= t_b).astype(int)

    after_a = fm.threshold_metrics(y_true[mask_a], pred_eq[mask_a])
    after_b = fm.threshold_metrics(y_true[mask_b], pred_eq[mask_b])

    gap_keys = ("sensitivity", "fnr", "specificity", "fpr",
                "ppv", "npv", "predicted_positive_rate")
    rows = []
    for k in gap_keys:
        before_gap = before_a[k] - before_b[k]
        after_gap  = after_a[k]  - after_b[k]
        rows.append({
            "metric":          k,
            "before_a":        before_a[k],
            "before_b":        before_b[k],
            "gap_before":      before_gap,
            "after_a":         after_a[k],
            "after_b":         after_b[k],
            "gap_after":       after_gap,
            "gap_change":      after_gap - before_gap,
        })
    df_out = pd.DataFrame(rows)

    # disparate impact ratio, before/after
    di_before = (before_a["predicted_positive_rate"] / before_b["predicted_positive_rate"]
                 if before_b["predicted_positive_rate"] else np.nan)
    di_after = (after_a["predicted_positive_rate"] / after_b["predicted_positive_rate"]
                if after_b["predicted_positive_rate"] else np.nan)

    return {
        "metric_equalized": metric_key,
        "target_value":     target_value,
        "shared_threshold": shared_threshold,
        "threshold_a":       t_a,
        "threshold_b":       t_b,
        "label_a":           label_a,
        "label_b":           label_b,
        "dir_before":        di_before,
        "dir_after":         di_after,
        "table":             df_out,
    }


# ---------------------------------------------------------------------------
# 19. Pareto frontier of fairness metrics under independent group thresholds
# ---------------------------------------------------------------------------

def pareto_frontier_grid(y_true, prob, g_test, group_a_val, group_b_val,
                          n_grid=25):
    """
    Sweep independent thresholds (t_a, t_b) for each subgroup over an
    n_grid x n_grid grid and compute, at every combination:
      - equal-opportunity violation   = |sensitivity gap|
      - demographic-parity violation  = |predicted-positive-rate gap|
      - PPV-parity violation          = |ppv gap|
      - NPV-parity violation          = |npv gap|
      - utility                       = pooled balanced accuracy
                                         (mean of sensitivity, specificity
                                         across both groups combined)

    Note: because thresholding alone cannot change calibration, this grid
    does not include a separate "calibration" axis (that trade-off is
    explored by the recalibration-method analysis, not threshold choice).

    Flags the Pareto-optimal subset (no other grid point is at least as
    good on all four violation axes and strictly better on one) and
    returns the full grid as a DataFrame with a `pareto_optimal` column.
    """
    from . import metrics as fm

    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)
    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val

    ya, pa = y_true[mask_a], prob[mask_a]
    yb, pb = y_true[mask_b], prob[mask_b]

    grid = np.linspace(0.01, 0.99, n_grid)

    rows = []
    for t_a in grid:
        pred_a = (pa >= t_a).astype(int)
        ma = fm.threshold_metrics(ya, pred_a)
        for t_b in grid:
            pred_b = (pb >= t_b).astype(int)
            mb = fm.threshold_metrics(yb, pred_b)

            eo_violation  = abs(ma["sensitivity"] - mb["sensitivity"])
            dp_violation  = abs(ma["predicted_positive_rate"] - mb["predicted_positive_rate"])
            ppv_violation = abs(ma["ppv"] - mb["ppv"]) if not (np.isnan(ma["ppv"]) or np.isnan(mb["ppv"])) else np.nan
            npv_violation = abs(ma["npv"] - mb["npv"]) if not (np.isnan(ma["npv"]) or np.isnan(mb["npv"])) else np.nan

            n_tot = ma["n"] + mb["n"]
            pooled_sens = (ma["sensitivity"] * ma["n_pos"] + mb["sensitivity"] * mb["n_pos"]) / \
                          (ma["n_pos"] + mb["n_pos"]) if (ma["n_pos"] + mb["n_pos"]) > 0 else np.nan
            pooled_spec = (ma["specificity"] * ma["n_neg"] + mb["specificity"] * mb["n_neg"]) / \
                          (ma["n_neg"] + mb["n_neg"]) if (ma["n_neg"] + mb["n_neg"]) > 0 else np.nan
            utility = np.nanmean([pooled_sens, pooled_spec])

            rows.append({
                "threshold_a":   float(t_a),
                "threshold_b":   float(t_b),
                "eo_violation":  eo_violation,
                "dp_violation":  dp_violation,
                "ppv_violation": ppv_violation,
                "npv_violation": npv_violation,
                "utility":       utility,
            })

    df = pd.DataFrame(rows).dropna(subset=["ppv_violation", "npv_violation"])

    # Pareto-optimality: minimize the 4 violation axes, maximize utility.
    obj_min = df[["eo_violation", "dp_violation", "ppv_violation", "npv_violation"]].values
    obj_max = df["utility"].values
    n = len(df)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        dominates = (
            (obj_min <= obj_min[i]).all(axis=1) &
            (obj_max >= obj_max[i]) &
            (
                (obj_min < obj_min[i]).any(axis=1) |
                (obj_max > obj_max[i])
            )
        )
        dominates[i] = False
        if dominates.any():
            is_optimal[i] = False

    df["pareto_optimal"] = is_optimal
    return df


# ---------------------------------------------------------------------------
# 13. Calibration-by-subgroup curves with bootstrap confidence bands
# ---------------------------------------------------------------------------

def calibration_curve_with_ci(y_true, prob, g_test, group_a_val, group_b_val,
                               label_a, label_b, n_bins=10, n_boot=300, seed=42):
    """
    Equal-width-bin calibration curve (mean predicted probability vs.
    observed event frequency) for each subgroup, with bootstrap 95% CIs
    on the observed frequency within each bin.

    Returns a tidy DataFrame: group, bin_low, bin_high, bin_mid,
    mean_predicted, n, observed_freq, ci_low, ci_high.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)
    rng    = np.random.default_rng(seed)

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for val, lbl in [(group_a_val, label_a), (group_b_val, label_b)]:
        mask = g_test == val
        yt, pr = y_true[mask], prob[mask]
        bin_idx = np.clip(np.digitize(pr, bins) - 1, 0, n_bins - 1)

        for b in range(n_bins):
            in_bin = bin_idx == b
            n_b = int(in_bin.sum())
            if n_b == 0:
                continue
            mean_pred = float(pr[in_bin].mean())
            obs_freq  = float(yt[in_bin].mean())

            idx_bin = np.where(in_bin)[0]
            boot = np.empty(n_boot)
            for i in range(n_boot):
                s = rng.choice(idx_bin, size=n_b, replace=True)
                boot[i] = yt[s].mean()
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

            rows.append({
                "group":          lbl,
                "bin_low":        float(bins[b]),
                "bin_high":       float(bins[b + 1]),
                "bin_mid":        float((bins[b] + bins[b + 1]) / 2),
                "mean_predicted": mean_pred,
                "n":              n_b,
                "observed_freq":  obs_freq,
                "ci_low":         float(ci_low),
                "ci_high":        float(ci_high),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 20 / 34. Calibration-method comparison (pooled vs. group-specific
# isotonic / Platt) and its use as a calibration-method robustness check
# ---------------------------------------------------------------------------

def fit_platt(y_true, prob):
    """Platt scaling: logistic regression of outcome on logit(probability)."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X, np.asarray(y_true, dtype=int))
    return model


def apply_platt(model, prob):
    eps = 1e-6
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(X)[:, 1]


def fit_isotonic_map(y_true, prob):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(np.asarray(prob, dtype=float), np.asarray(y_true, dtype=float))
    return ir


def recalibration_strategies(y_tr, prob_tr_uncal, g_tr,
                              prob_te_uncal, g_te,
                              group_a_val, group_b_val):
    """
    Fit 5 recalibration strategies on the training-set base-model scores
    and apply them to the test-set base-model scores:
      1. uncalibrated            -- raw base-model probability
      2. pooled_isotonic         -- one isotonic map fit on all training data
      3. pooled_platt            -- one Platt (logistic-on-logit) map
      4. group_isotonic          -- separate isotonic map per subgroup
      5. group_platt             -- separate Platt map per subgroup

    Returns {strategy_name: recalibrated test-set probability array}.
    """
    y_tr          = np.asarray(y_tr, dtype=int)
    prob_tr_uncal = np.asarray(prob_tr_uncal, dtype=float)
    g_tr          = np.asarray(g_tr)
    prob_te_uncal = np.asarray(prob_te_uncal, dtype=float)
    g_te          = np.asarray(g_te)

    out = {"uncalibrated": prob_te_uncal.copy()}

    pooled_iso = fit_isotonic_map(y_tr, prob_tr_uncal)
    out["pooled_isotonic"] = pooled_iso.predict(prob_te_uncal)

    pooled_platt = fit_platt(y_tr, prob_tr_uncal)
    out["pooled_platt"] = apply_platt(pooled_platt, prob_te_uncal)

    group_iso_pred = np.full_like(prob_te_uncal, np.nan, dtype=float)
    group_platt_pred = np.full_like(prob_te_uncal, np.nan, dtype=float)
    for val in (group_a_val, group_b_val):
        tr_mask = g_tr == val
        te_mask = g_te == val
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        iso_g = fit_isotonic_map(y_tr[tr_mask], prob_tr_uncal[tr_mask])
        group_iso_pred[te_mask] = iso_g.predict(prob_te_uncal[te_mask])

        platt_g = fit_platt(y_tr[tr_mask], prob_tr_uncal[tr_mask])
        group_platt_pred[te_mask] = apply_platt(platt_g, prob_te_uncal[te_mask])

    out["group_isotonic"] = group_iso_pred
    out["group_platt"] = group_platt_pred
    return out


def calibration_strategy_table(y_te, g_te, group_a_val, group_b_val,
                                label_a, label_b, strategies, threshold=0.5):
    """
    For each recalibration strategy (dict from recalibration_strategies),
    recompute per-subgroup discrimination/calibration metrics and the raw
    + prevalence-adjusted PPV/NPV gaps, plus the sensitivity/FNR/PPR gaps.

    Decision threshold defaults to 0.5, the standard convention once scores
    have been recalibrated to be probabilistically meaningful.

    Returns a tidy DataFrame: one row per (strategy, subgroup), plus a
    companion gap table accessible via the second return value.
    """
    from . import metrics as fm
    from .adjustments import ppv_npv_at_prevalence, attenuation_pct

    y_te = np.asarray(y_te, dtype=int)
    g_te = np.asarray(g_te)
    mask_a = g_te == group_a_val
    mask_b = g_te == group_b_val

    subgroup_rows = []
    gap_rows = []

    for strat_name, prob_te in strategies.items():
        prob_te = np.asarray(prob_te, dtype=float)
        valid = ~np.isnan(prob_te)
        pred_te = (prob_te >= threshold).astype(float)
        pred_te[~valid] = np.nan

        ma = fm.subgroup_metrics(y_te[mask_a & valid], prob_te[mask_a & valid],
                                  pred_te[mask_a & valid].astype(int))
        mb = fm.subgroup_metrics(y_te[mask_b & valid], prob_te[mask_b & valid],
                                  pred_te[mask_b & valid].astype(int))

        for lbl, m in [(label_a, ma), (label_b, mb)]:
            subgroup_rows.append({
                "strategy": strat_name, "group": lbl,
                "n": m["n"], "auroc": m["auroc"], "auprc": m["auprc"],
                "brier": m["brier"], "ece": m["ece"],
                "calib_intercept": m["calib_intercept"], "calib_slope": m["calib_slope"],
                "sensitivity": m["sensitivity"], "fnr": m["fnr"],
                "specificity": m["specificity"], "fpr": m["fpr"],
                "ppv": m["ppv"], "npv": m["npv"],
                "predicted_positive_rate": m["predicted_positive_rate"],
            })

        target_prev = (ma["n_pos"] + mb["n_pos"]) / (ma["n"] + mb["n"])
        adj_a = ppv_npv_at_prevalence(ma["sensitivity"], ma["specificity"], target_prev)
        adj_b = ppv_npv_at_prevalence(mb["sensitivity"], mb["specificity"], target_prev)

        sens_gap = ma["sensitivity"] - mb["sensitivity"]
        fnr_gap  = ma["fnr"] - mb["fnr"]
        ppr_gap  = ma["predicted_positive_rate"] - mb["predicted_positive_rate"]
        ppv_gap_raw = ma["ppv"] - mb["ppv"]
        npv_gap_raw = ma["npv"] - mb["npv"]
        ppv_gap_adj = adj_a["ppv"] - adj_b["ppv"]
        npv_gap_adj = adj_a["npv"] - adj_b["npv"]
        di = (ma["predicted_positive_rate"] / mb["predicted_positive_rate"]
              if mb["predicted_positive_rate"] else np.nan)

        gap_rows.append({
            "strategy":          strat_name,
            "auroc_A":           ma["auroc"], "auroc_B": mb["auroc"],
            "ece_A":             ma["ece"],   "ece_B":   mb["ece"],
            "brier_A":           ma["brier"], "brier_B": mb["brier"],
            "sensitivity_gap":   sens_gap,
            "fnr_gap":           fnr_gap,
            "ppr_gap":           ppr_gap,
            "disparate_impact_ratio": di,
            "ppv_gap_raw":       ppv_gap_raw,
            "ppv_gap_adj":       ppv_gap_adj,
            "ppv_attenuation_pct": attenuation_pct(ppv_gap_raw, ppv_gap_adj),
            "npv_gap_raw":       npv_gap_raw,
            "npv_gap_adj":       npv_gap_adj,
            "npv_attenuation_pct": attenuation_pct(npv_gap_raw, npv_gap_adj),
        })

    return pd.DataFrame(subgroup_rows), pd.DataFrame(gap_rows)


# ---------------------------------------------------------------------------
# 48. Protected-label permutation test
# ---------------------------------------------------------------------------

def protected_label_permutation_test(y_true, pred, prob, g_test,
                                      group_a_val, group_b_val,
                                      n_perm=1000, seed=42):
    """
    Shuffle the protected-attribute labels (sex) across the test set,
    holding outcomes, predictions, and scores fixed, and rebuild the null
    distribution of each fairness gap under "no real subgroup structure".

    Returns (summary_df, null_df):
      summary_df -- one row per metric: observed gap, null mean/SD,
                    null 95% interval, empirical two-sided p-value
      null_df    -- long-format null distribution (n_perm rows per metric)
                    for plotting
    """
    y_true = np.asarray(y_true, dtype=int)
    pred   = np.asarray(pred,   dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    g_test = np.asarray(g_test)
    rng    = np.random.default_rng(seed)

    n_a = int((g_test == group_a_val).sum())
    metric_keys = ("sensitivity", "fnr", "ppv", "npv", "predicted_positive_rate")

    def _gaps(mask_a, mask_b):
        ma = fm.threshold_metrics(y_true[mask_a], pred[mask_a])
        mb = fm.threshold_metrics(y_true[mask_b], pred[mask_b])
        return {k: ma[k] - mb[k] for k in metric_keys}

    mask_a_obs = g_test == group_a_val
    mask_b_obs = g_test == group_b_val
    observed = _gaps(mask_a_obs, mask_b_obs)

    n = len(g_test)
    null_vals = {k: np.empty(n_perm) for k in metric_keys}
    idx_all = np.arange(n)
    for i in range(n_perm):
        perm_idx = rng.permutation(idx_all)
        mask_a_perm = np.zeros(n, dtype=bool)
        mask_a_perm[perm_idx[:n_a]] = True
        mask_b_perm = ~mask_a_perm
        g = _gaps(mask_a_perm, mask_b_perm)
        for k in metric_keys:
            null_vals[k][i] = g[k]

    summary_rows = []
    null_rows = []
    for k in metric_keys:
        null = null_vals[k]
        obs = observed[k]
        p_emp = float(np.mean(np.abs(null) >= abs(obs))) if not np.isnan(obs) else np.nan
        summary_rows.append({
            "metric":          k,
            "observed_gap":    obs,
            "null_mean":       float(np.nanmean(null)),
            "null_sd":         float(np.nanstd(null)),
            "null_ci_low":     float(np.nanquantile(null, 0.025)),
            "null_ci_high":    float(np.nanquantile(null, 0.975)),
            "empirical_p":     p_emp,
            "n_perm":          n_perm,
        })
        for v in null:
            null_rows.append({"metric": k, "null_gap": v})

    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


# ---------------------------------------------------------------------------
# 49. Outcome-label permutation test
# ---------------------------------------------------------------------------

def outcome_label_permutation_test(X_train, y_train, X_test, y_test, g_test,
                                    group_a_val, group_b_val,
                                    n_perm=200, seed=42, threshold=0.5):
    """
    Shuffle the *training-set* outcome labels (breaking the true
    relationship between features and outcome), refit a fast logistic
    regression on each shuffled copy, and evaluate AUROC and fairness
    gaps on the (unshuffled) test set.

    If the original pipeline is detecting real predictive / fairness
    structure rather than an artifact, AUROC should collapse to ~0.5 and
    fairness gaps should lose any stable, structured pattern once the
    outcome label is meaningless.

    n_perm defaults to 200 (not 1000) because each iteration refits a
    model; 200 refits already gives a stable null distribution for a
    single fast linear model and keeps runtime tractable across 5
    datasets.

    Returns (summary_df, null_df) in the same shape as
    protected_label_permutation_test, plus an "auroc" pseudo-metric.
    """
    from sklearn.metrics import roc_auc_score

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test,  dtype=int)
    g_test  = np.asarray(g_test)
    rng     = np.random.default_rng(seed)

    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val
    metric_keys = ("sensitivity", "fnr", "ppv", "npv", "predicted_positive_rate")

    null_auroc = np.empty(n_perm)
    null_vals  = {k: np.empty(n_perm) for k in metric_keys}

    for i in range(n_perm):
        y_shuf = rng.permutation(y_train)
        try:
            clf = LogisticRegression(max_iter=300)
            clf.fit(X_train, y_shuf)
            prob_te = clf.predict_proba(X_test)[:, 1]
        except Exception:
            null_auroc[i] = np.nan
            for k in metric_keys:
                null_vals[k][i] = np.nan
            continue

        pred_te = (prob_te >= threshold).astype(int)
        try:
            null_auroc[i] = roc_auc_score(y_test, prob_te)
        except Exception:
            null_auroc[i] = np.nan

        ma = fm.threshold_metrics(y_test[mask_a], pred_te[mask_a])
        mb = fm.threshold_metrics(y_test[mask_b], pred_te[mask_b])
        for k in metric_keys:
            null_vals[k][i] = ma[k] - mb[k]

    summary_rows = [{
        "metric":       "auroc",
        "observed_gap": np.nan,
        "null_mean":    float(np.nanmean(null_auroc)),
        "null_sd":      float(np.nanstd(null_auroc)),
        "null_ci_low":  float(np.nanquantile(null_auroc, 0.025)),
        "null_ci_high": float(np.nanquantile(null_auroc, 0.975)),
        "empirical_p":  np.nan,
        "n_perm":       n_perm,
    }]
    null_rows = [{"metric": "auroc", "null_gap": v} for v in null_auroc]

    for k in metric_keys:
        null = null_vals[k]
        summary_rows.append({
            "metric":       k,
            "observed_gap": np.nan,   # filled in by caller with the real fitted model's gap
            "null_mean":    float(np.nanmean(null)),
            "null_sd":      float(np.nanstd(null)),
            "null_ci_low":  float(np.nanquantile(null, 0.025)),
            "null_ci_high": float(np.nanquantile(null, 0.975)),
            "empirical_p":  np.nan,
            "n_perm":       n_perm,
        })
        for v in null:
            null_rows.append({"metric": k, "null_gap": v})

    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


# ---------------------------------------------------------------------------
# 50. Null-feature model comparison
# ---------------------------------------------------------------------------

def null_feature_model_comparison(X_train, y_train, X_test, y_test, g_test,
                                   group_a_val, group_b_val,
                                   age_col_idx=None, clinical_block_idx=None,
                                   seed=42, threshold=0.5):
    """
    Fit a ladder of trivial baseline models on the real (unshuffled) data
    and compare their discrimination and fairness-gap profile:
      - intercept_only      : predicts the pooled training prevalence for
                               everyone (no features)
      - age_only            : single-feature logistic regression on age,
                               if an age column is identifiable
      - random_score        : uniform random score, U(0, 1)
      - simple_clinical_baseline : logistic regression on a small
                               clinically-motivated covariate block (e.g.
                               comorbidities), if identifiable

    Returns a DataFrame with one row per model: AUROC, AUPRC, Brier, ECE,
    sensitivity/FNR/PPV/NPV/PPR gaps. The caller appends the full ML
    model's own row (already computed by the main pipeline) for
    comparison.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test  = np.asarray(X_test,  dtype=float)
    y_test  = np.asarray(y_test,  dtype=int)
    g_test  = np.asarray(g_test)
    rng     = np.random.default_rng(seed)

    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val

    def _row(model_name, prob_te):
        prob_te = np.asarray(prob_te, dtype=float)
        pred_te = (prob_te >= threshold).astype(int)
        try:
            auroc = roc_auc_score(y_test, prob_te)
        except Exception:
            auroc = np.nan
        try:
            auprc = average_precision_score(y_test, prob_te)
        except Exception:
            auprc = np.nan
        cal = fm.calibration_metrics(y_test, prob_te)

        ma = fm.threshold_metrics(y_test[mask_a], pred_te[mask_a])
        mb = fm.threshold_metrics(y_test[mask_b], pred_te[mask_b])

        return {
            "model":            model_name,
            "auroc":            auroc,
            "auprc":            auprc,
            "brier":            cal["brier"],
            "ece":              cal["ece"],
            "sensitivity_gap":  ma["sensitivity"] - mb["sensitivity"],
            "fnr_gap":          ma["fnr"] - mb["fnr"],
            "ppv_gap":          ma["ppv"] - mb["ppv"],
            "npv_gap":          ma["npv"] - mb["npv"],
            "ppr_gap":          ma["predicted_positive_rate"] - mb["predicted_positive_rate"],
        }

    rows = []

    # intercept-only
    pooled_prev = float(y_train.mean())
    prob_intercept = np.full(len(y_test), pooled_prev)
    rows.append(_row("intercept_only", prob_intercept))

    # age-only
    if age_col_idx is not None:
        try:
            clf = LogisticRegression(max_iter=300)
            clf.fit(X_train[:, [age_col_idx]], y_train)
            prob_age = clf.predict_proba(X_test[:, [age_col_idx]])[:, 1]
            rows.append(_row("age_only", prob_age))
        except Exception:
            pass

    # random score
    prob_random = rng.random(len(y_test))
    rows.append(_row("random_score", prob_random))

    # simple clinical baseline
    if clinical_block_idx is not None and len(clinical_block_idx) > 0:
        try:
            clf = LogisticRegression(max_iter=300)
            clf.fit(X_train[:, clinical_block_idx], y_train)
            prob_clin = clf.predict_proba(X_test[:, clinical_block_idx])[:, 1]
            rows.append(_row("simple_clinical_baseline", prob_clin))
        except Exception:
            pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared helper: fit + isotonic-calibrate a model exactly as pipeline.py does
# ---------------------------------------------------------------------------

def _fit_and_calibrate(model, X_train, y_train, X_test, cv=5):
    """Fit and fold-ensemble-calibrate exactly as the primary pipeline does."""
    from .pipeline import _make_calibrated_classifier
    calibrated = _make_calibrated_classifier(model, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated.predict_proba(X_test)[:, 1]


def _gap_row(y_test, prob_te, mask_a, mask_b, threshold=0.5, extra=None):
    """Compute one row of AUROC/AUPRC/calibration + raw & prevalence-adjusted
    fairness gaps for a given test-set probability vector."""
    y_test = np.asarray(y_test, dtype=int)
    prob_te = np.asarray(prob_te, dtype=float)
    pred_te = (prob_te >= threshold).astype(int)

    ma = fm.subgroup_metrics(y_test[mask_a], prob_te[mask_a], pred_te[mask_a])
    mb = fm.subgroup_metrics(y_test[mask_b], prob_te[mask_b], pred_te[mask_b])
    pooled = fm.subgroup_metrics(y_test, prob_te, pred_te)

    target_prev = (ma["n_pos"] + mb["n_pos"]) / (ma["n"] + mb["n"])
    adj_a = ppv_npv_at_prevalence(ma["sensitivity"], ma["specificity"], target_prev)
    adj_b = ppv_npv_at_prevalence(mb["sensitivity"], mb["specificity"], target_prev)

    ppv_gap_raw = ma["ppv"] - mb["ppv"]
    npv_gap_raw = ma["npv"] - mb["npv"]
    ppv_gap_adj = adj_a["ppv"] - adj_b["ppv"]
    npv_gap_adj = adj_a["npv"] - adj_b["npv"]

    row = {
        "auroc":             pooled["auroc"],
        "auprc":             pooled["auprc"],
        "brier":             pooled["brier"],
        "ece":               pooled["ece"],
        "sensitivity_A":     ma["sensitivity"], "sensitivity_B": mb["sensitivity"],
        "sensitivity_gap":   ma["sensitivity"] - mb["sensitivity"],
        "fnr_gap":           ma["fnr"] - mb["fnr"],
        "specificity_gap":   ma["specificity"] - mb["specificity"],
        "fpr_gap":           ma["fpr"] - mb["fpr"],
        "ppr_gap":           ma["predicted_positive_rate"] - mb["predicted_positive_rate"],
        "disparate_impact_ratio": (ma["predicted_positive_rate"] / mb["predicted_positive_rate"]
                                    if mb["predicted_positive_rate"] else np.nan),
        "ppv_gap_raw":       ppv_gap_raw,
        "ppv_gap_adj":       ppv_gap_adj,
        "ppv_attenuation_pct": attenuation_pct(ppv_gap_raw, ppv_gap_adj),
        "npv_gap_raw":       npv_gap_raw,
        "npv_gap_adj":       npv_gap_adj,
        "npv_attenuation_pct": attenuation_pct(npv_gap_raw, npv_gap_adj),
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 32. Model-family robustness
# ---------------------------------------------------------------------------

def model_family_robustness(X_train, y_train, X_test, y_test, mask_a, mask_b,
                             random_state=42, threshold=0.5):
    """
    Refit the entire decomposition (AUROC/AUPRC, raw gaps, prevalence-adjusted
    gaps) under several model families, each isotonic-calibrated the same
    way as the main pipeline.  Tests whether the need for decomposition --
    and the empirical sex-gap pattern -- is model-independent.

    Families are the single canonical pool defined by
    `pipeline._build_models` — the same six families used for primary model
    selection — so the selection pool and this robustness pool are identical:
    logistic regression, L1-penalized logistic regression, random forest, a
    shallow decision tree, XGBoost (if installed), and a soft-voting ensemble.

    Returns a DataFrame with one row per model family.
    """
    from .pipeline import _build_models

    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test  = np.asarray(y_test,  dtype=int)

    families = _build_models(random_state)

    rows = []
    for fam_name, model in families.items():
        try:
            prob_te = _fit_and_calibrate(model, X_train, y_train, X_test)
            row = _gap_row(y_test, prob_te, mask_a, mask_b, threshold=threshold,
                            extra={"model_family": fam_name})
            rows.append(row)
        except Exception as e:
            rows.append({"model_family": fam_name, "error": str(e)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 35. Threshold-policy robustness
# ---------------------------------------------------------------------------

def derive_threshold(y_tr, prob_tr, policy, **kwargs):
    """
    Derive a single shared decision threshold from TRAINING-set scores
    (never the test set), following one of several common policies:

      "fixed_050"        -- 0.50
      "prevalence"        -- training-set outcome prevalence
      "youden"             -- maximizes sensitivity + specificity - 1
      "fixed_sensitivity"  -- smallest threshold achieving a target
                              sensitivity (kwargs: target=0.80)
      "fixed_specificity"  -- smallest threshold achieving a target
                              specificity (kwargs: target=0.80)
      "top_k"               -- flags the top k fraction of training scores
                              as positive (kwargs: k=0.20)
      "cost_sensitive"      -- minimizes total cost on a grid of thresholds
                              (kwargs: cost_fn=5, cost_fp=1)
    """
    from sklearn.metrics import roc_curve

    y_tr   = np.asarray(y_tr, dtype=int)
    prob_tr = np.asarray(prob_tr, dtype=float)

    if policy == "fixed_050":
        return 0.5
    if policy == "prevalence":
        return float(y_tr.mean())
    if policy == "youden":
        fpr, tpr, thr = roc_curve(y_tr, prob_tr)
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    if policy == "fixed_sensitivity":
        return find_threshold_for_metric(y_tr, prob_tr, "sensitivity",
                                          kwargs.get("target", 0.80))
    if policy == "fixed_specificity":
        return find_threshold_for_metric(y_tr, prob_tr, "specificity",
                                          kwargs.get("target", 0.80))
    if policy == "top_k":
        k = kwargs.get("k", 0.20)
        return float(np.quantile(prob_tr, 1 - k))
    if policy == "cost_sensitive":
        cost_fn = kwargs.get("cost_fn", 5.0)
        cost_fp = kwargs.get("cost_fp", 1.0)
        grid = np.linspace(0.01, 0.99, 199)
        best_t, best_cost = 0.5, np.inf
        for t in grid:
            pred = (prob_tr >= t).astype(int)
            fn = int(((y_tr == 1) & (pred == 0)).sum())
            fp = int(((y_tr == 0) & (pred == 1)).sum())
            cost = cost_fn * fn + cost_fp * fp
            if cost < best_cost:
                best_cost, best_t = cost, float(t)
        return best_t
    raise ValueError(f"Unknown threshold policy: {policy}")


def threshold_policy_robustness(y_train, prob_train, y_test, prob_test,
                                 mask_a, mask_b):
    """
    Apply 7 threshold policies (derived from TRAINING-set scores) to the
    same fixed test-set probabilities, and recompute fairness gaps under
    each.  Tests whether the decomposition conclusion depends on one
    threshold convention.

    Returns a DataFrame with one row per policy.
    """
    policies = [
        ("fixed_050",        {}),
        ("prevalence",       {}),
        ("youden",           {}),
        ("fixed_sensitivity",{"target": 0.80}),
        ("fixed_specificity",{"target": 0.80}),
        ("top_k",            {"k": 0.20}),
        ("cost_sensitive",   {"cost_fn": 5.0, "cost_fp": 1.0}),
    ]
    rows = []
    for policy, kwargs in policies:
        try:
            t = derive_threshold(y_train, prob_train, policy, **kwargs)
            row = _gap_row(y_test, prob_test, mask_a, mask_b, threshold=t,
                            extra={"policy": policy, "threshold": t})
            rows.append(row)
        except Exception as e:
            rows.append({"policy": policy, "error": str(e)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 36. Protected-attribute inclusion analysis
# ---------------------------------------------------------------------------

def protected_attribute_inclusion_analysis(X_train, y_train, g_train,
                                            X_test, y_test, g_test,
                                            group_a_val, group_b_val,
                                            random_state=42, threshold=0.5,
                                            target_sensitivity=None):
    """
    Compare 5 modeling strategies for handling the protected attribute,
    all using logistic regression as the base learner for a clean,
    apples-to-apples comparison (tree ensembles handle a one-hot sex
    feature inconsistently across libraries, which would confound the
    comparison):

      1. excludes_sex                   -- sex not in the feature set
      2. includes_sex                   -- sex added as a binary feature
      3. excludes_sex_group_calibration -- (1)'s model, then group-specific
                                            isotonic recalibration
      4. excludes_sex_group_threshold   -- (1)'s model, with per-group
                                            thresholds equalizing sensitivity
      5. subgroup_specific_models       -- two entirely separate models,
                                            one fit per subgroup

    Returns a DataFrame with one row per strategy.
    """
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test  = np.asarray(y_test,  dtype=int)
    g_train = np.asarray(g_train)
    g_test  = np.asarray(g_test)

    mask_a_tr, mask_b_tr = g_train == group_a_val, g_train == group_b_val
    mask_a_te, mask_b_te = g_test == group_a_val,  g_test == group_b_val

    def _fit_lr(Xtr, ytr):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        clf = LR(max_iter=2000, random_state=random_state)
        clf.fit(Xtr_s, ytr)
        return scaler, clf

    rows = []

    # 1. excludes sex
    scaler1, clf1 = _fit_lr(X_train, y_train)
    prob1_tr = clf1.predict_proba(scaler1.transform(X_train))[:, 1]
    prob1_te = clf1.predict_proba(scaler1.transform(X_test))[:, 1]
    rows.append(_gap_row(y_test, prob1_te, mask_a_te, mask_b_te, threshold=threshold,
                          extra={"strategy": "excludes_sex"}))

    # 2. includes sex (binary indicator appended as last column)
    sex_tr = (g_train == group_a_val).astype(float).reshape(-1, 1)
    sex_te = (g_test == group_a_val).astype(float).reshape(-1, 1)
    X_train_sex = np.hstack([X_train, sex_tr])
    X_test_sex  = np.hstack([X_test, sex_te])
    scaler2, clf2 = _fit_lr(X_train_sex, y_train)
    prob2_te = clf2.predict_proba(scaler2.transform(X_test_sex))[:, 1]
    rows.append(_gap_row(y_test, prob2_te, mask_a_te, mask_b_te, threshold=threshold,
                          extra={"strategy": "includes_sex"}))

    # 3. excludes sex, but group-specific isotonic recalibration on top of (1)
    iso_a = fit_isotonic_map(y_train[mask_a_tr], prob1_tr[mask_a_tr])
    iso_b = fit_isotonic_map(y_train[mask_b_tr], prob1_tr[mask_b_tr])
    prob3_te = np.empty_like(prob1_te)
    prob3_te[mask_a_te] = iso_a.predict(prob1_te[mask_a_te])
    prob3_te[mask_b_te] = iso_b.predict(prob1_te[mask_b_te])
    rows.append(_gap_row(y_test, prob3_te, mask_a_te, mask_b_te, threshold=threshold,
                          extra={"strategy": "excludes_sex_group_calibration"}))

    # 4. excludes sex, but group-specific thresholds equalizing sensitivity
    from .adjustments import equal_sensitivity_thresholds
    target_sens = target_sensitivity
    if target_sens is None:
        target_sens = fm.threshold_metrics(y_train, (prob1_tr >= threshold).astype(int))["sensitivity"]
    eq_thr = equal_sensitivity_thresholds(
        y_train, prob1_tr, g_train, [group_a_val, group_b_val], target_sens
    )
    pred4_te = np.zeros(len(y_test), dtype=int)
    pred4_te[mask_a_te] = (prob1_te[mask_a_te] >= eq_thr[group_a_val]).astype(int)
    pred4_te[mask_b_te] = (prob1_te[mask_b_te] >= eq_thr[group_b_val]).astype(int)
    ma4 = fm.subgroup_metrics(y_test[mask_a_te], prob1_te[mask_a_te], pred4_te[mask_a_te])
    mb4 = fm.subgroup_metrics(y_test[mask_b_te], prob1_te[mask_b_te], pred4_te[mask_b_te])
    target_prev4 = (ma4["n_pos"] + mb4["n_pos"]) / (ma4["n"] + mb4["n"])
    adj_a4 = ppv_npv_at_prevalence(ma4["sensitivity"], ma4["specificity"], target_prev4)
    adj_b4 = ppv_npv_at_prevalence(mb4["sensitivity"], mb4["specificity"], target_prev4)
    ppv_gap_raw4 = ma4["ppv"] - mb4["ppv"]
    npv_gap_raw4 = ma4["npv"] - mb4["npv"]
    ppv_gap_adj4 = adj_a4["ppv"] - adj_b4["ppv"]
    npv_gap_adj4 = adj_a4["npv"] - adj_b4["npv"]
    rows.append({
        "strategy":          "excludes_sex_group_threshold",
        "auroc":             fm.discrimination_metrics(y_test, prob1_te)["auroc"],
        "auprc":             fm.discrimination_metrics(y_test, prob1_te)["auprc"],
        "brier":             np.nan, "ece": np.nan,
        "sensitivity_A":     ma4["sensitivity"], "sensitivity_B": mb4["sensitivity"],
        "sensitivity_gap":   ma4["sensitivity"] - mb4["sensitivity"],
        "fnr_gap":           ma4["fnr"] - mb4["fnr"],
        "specificity_gap":   ma4["specificity"] - mb4["specificity"],
        "fpr_gap":           ma4["fpr"] - mb4["fpr"],
        "ppr_gap":           ma4["predicted_positive_rate"] - mb4["predicted_positive_rate"],
        "disparate_impact_ratio": (ma4["predicted_positive_rate"] / mb4["predicted_positive_rate"]
                                    if mb4["predicted_positive_rate"] else np.nan),
        "ppv_gap_raw":       ppv_gap_raw4, "ppv_gap_adj": ppv_gap_adj4,
        "ppv_attenuation_pct": attenuation_pct(ppv_gap_raw4, ppv_gap_adj4),
        "npv_gap_raw":       npv_gap_raw4, "npv_gap_adj": npv_gap_adj4,
        "npv_attenuation_pct": attenuation_pct(npv_gap_raw4, npv_gap_adj4),
        "threshold_a":       eq_thr[group_a_val], "threshold_b": eq_thr[group_b_val],
    })

    # 5. subgroup-specific models (fit separately on each group's training data)
    scaler_a, clf_a = _fit_lr(X_train[mask_a_tr], y_train[mask_a_tr])
    scaler_b, clf_b = _fit_lr(X_train[mask_b_tr], y_train[mask_b_tr])
    prob5_te = np.empty(len(y_test), dtype=float)
    prob5_te[mask_a_te] = clf_a.predict_proba(scaler_a.transform(X_test[mask_a_te]))[:, 1]
    prob5_te[mask_b_te] = clf_b.predict_proba(scaler_b.transform(X_test[mask_b_te]))[:, 1]
    rows.append(_gap_row(y_test, prob5_te, mask_a_te, mask_b_te, threshold=threshold,
                          extra={"strategy": "subgroup_specific_models"}))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 42. Random-effects meta-analysis with leave-one-out
# ---------------------------------------------------------------------------

def _ci_to_se(ci_low, ci_high, z=1.959963984540054):
    """Back out a normal-approximation SE from a symmetric 95% CI."""
    return (ci_high - ci_low) / (2 * z)


def random_effects_meta(dataset_names, estimates, ses, ci_z=1.959963984540054):
    """
    DerSimonian-Laird random-effects meta-analysis of a single gap (e.g. the
    sensitivity gap) across datasets, with heterogeneity (Q, tau^2, I^2),
    a 95% prediction interval, and leave-one-dataset-out re-pooling.

    Returns a dict:
      pooled_estimate, pooled_se, pooled_ci_low/high,
      tau2, Q, Q_df, Q_pvalue, I2,
      prediction_interval_low/high,
      per_study (DataFrame: dataset, estimate, se, weight_fixed_pct,
        weight_random_pct),
      leave_one_out (DataFrame: excluded_dataset, pooled_estimate, ci_low, ci_high)
    """
    from scipy import stats as scipy_stats

    dataset_names = list(dataset_names)
    estimates = np.asarray(estimates, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(estimates)
    variances = ses ** 2

    def _dl_pool(est, var):
        w_fixed = 1.0 / var
        pooled_fixed = np.sum(w_fixed * est) / np.sum(w_fixed)
        Q = np.sum(w_fixed * (est - pooled_fixed) ** 2)
        df = len(est) - 1
        C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
        w_rand = 1.0 / (var + tau2)
        pooled = np.sum(w_rand * est) / np.sum(w_rand)
        pooled_se = np.sqrt(1.0 / np.sum(w_rand))
        return pooled, pooled_se, tau2, Q, df, w_fixed, w_rand

    pooled, pooled_se, tau2, Q, df, w_fixed, w_rand = _dl_pool(estimates, variances)
    Q_p = float(1 - scipy_stats.chi2.cdf(Q, df)) if df > 0 else np.nan
    I2 = max(0.0, (Q - df) / Q) * 100 if (df > 0 and Q > 0) else 0.0

    pooled_ci_low = pooled - ci_z * pooled_se
    pooled_ci_high = pooled + ci_z * pooled_se

    # 95% prediction interval (accounts for between-study heterogeneity tau2;
    # uses a t-distribution with k-2 df, the standard small-k approximation)
    if k > 2:
        t_crit = float(scipy_stats.t.ppf(0.975, k - 2))
        pred_se = np.sqrt(pooled_se ** 2 + tau2)
        pred_low = pooled - t_crit * pred_se
        pred_high = pooled + t_crit * pred_se
    else:
        pred_low, pred_high = np.nan, np.nan

    per_study = pd.DataFrame({
        "dataset":      dataset_names,
        "estimate":     estimates,
        "se":           ses,
        "weight_fixed_pct":  100 * w_fixed / w_fixed.sum(),
        "weight_random_pct": 100 * w_rand / w_rand.sum(),
    })

    loo_rows = []
    for i in range(k):
        keep = np.arange(k) != i
        if keep.sum() < 2:
            continue
        p_i, se_i, tau2_i, Q_i, df_i, _, _ = _dl_pool(estimates[keep], variances[keep])
        loo_rows.append({
            "excluded_dataset": dataset_names[i],
            "pooled_estimate":  p_i,
            "pooled_se":        se_i,
            "ci_low":           p_i - ci_z * se_i,
            "ci_high":          p_i + ci_z * se_i,
            "tau2":             tau2_i,
        })
    leave_one_out = pd.DataFrame(loo_rows)

    return {
        "pooled_estimate":  pooled,
        "pooled_se":        pooled_se,
        "pooled_ci_low":    pooled_ci_low,
        "pooled_ci_high":   pooled_ci_high,
        "tau2":             tau2,
        "Q":                Q,
        "Q_df":             df,
        "Q_pvalue":         Q_p,
        "I2":               I2,
        "prediction_interval_low":  pred_low,
        "prediction_interval_high": pred_high,
        "per_study":        per_study,
        "leave_one_out":    leave_one_out,
    }


# ---------------------------------------------------------------------------
# 44. Multiple-comparison correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------------

def bh_fdr_correction(df, p_col="p_value", alpha=0.05):
    """
    Benjamini-Hochberg FDR correction on a long-format DataFrame of
    p-values (one row per test, any number of identifying columns plus
    `p_col`).

    Adds columns: rank, bh_critical_value, q_value (BH-adjusted p, i.e. the
    smallest alpha at which this test would survive correction), and
    `significant_bh` (True/False at the given `alpha`).
    """
    out = df.copy().reset_index(drop=True)
    valid = out[p_col].notna()
    n = int(valid.sum())

    out["rank"] = np.nan
    out["bh_critical_value"] = np.nan
    out["q_value"] = np.nan
    out["significant_bh"] = False

    if n == 0:
        return out

    sub = out.loc[valid].sort_values(p_col).copy()
    sub["rank"] = np.arange(1, n + 1)
    sub["bh_critical_value"] = sub["rank"] / n * alpha

    # q-value: smallest alpha at which each test (in ascending-p order)
    # would be rejected, enforcing monotonicity from the largest p down.
    p_sorted = sub[p_col].values
    ranks = sub["rank"].values
    raw_q = p_sorted * n / ranks
    q_monotone = np.minimum.accumulate(raw_q[::-1])[::-1]
    sub["q_value"] = np.clip(q_monotone, 0, 1)
    sub["significant_bh"] = sub[p_col] <= sub["bh_critical_value"]

    out.loc[sub.index, ["rank", "bh_critical_value", "q_value", "significant_bh"]] = \
        sub[["rank", "bh_critical_value", "q_value", "significant_bh"]]

    return out


# ---------------------------------------------------------------------------
# 45. Equivalence tests ("no meaningful gap") via TOST
# ---------------------------------------------------------------------------

def equivalence_test(estimate, ci_low, ci_high, margin=0.02, ci_level=0.95):
    """
    Two-one-sided-tests (TOST) equivalence test: is `estimate` statistically
    indistinguishable from zero within +/- `margin`?

    Backs out the SE from the supplied (ci_low, ci_high) at `ci_level`,
    builds the tighter 90% CI appropriate for a two-sided-alpha=0.10 TOST
    at the conventional 0.05-per-side test, and checks whether that CI
    falls entirely inside (-margin, +margin).

    Returns dict: estimate, se, tost_ci_low, tost_ci_high, margin,
    equivalent (bool), interpretation (str).
    """
    from scipy import stats as scipy_stats

    z_ci = scipy_stats.norm.ppf(0.5 + ci_level / 2)
    se = (ci_high - ci_low) / (2 * z_ci) if not (np.isnan(ci_low) or np.isnan(ci_high)) else np.nan

    z_tost = scipy_stats.norm.ppf(0.95)  # one-sided 5% -> 90% CI
    tost_low = estimate - z_tost * se if not np.isnan(se) else np.nan
    tost_high = estimate + z_tost * se if not np.isnan(se) else np.nan

    if np.isnan(se):
        equivalent = False
        interp = "insufficient information (no SE)"
    elif tost_low > -margin and tost_high < margin:
        equivalent = True
        interp = f"statistically equivalent to zero within +/-{margin}"
    elif abs(estimate) <= 1e-12 and (ci_high - ci_low) > 2 * margin:
        equivalent = False
        interp = "non-significant but underpowered -- not equivalence"
    else:
        equivalent = False
        interp = f"not equivalent to zero within +/-{margin}"

    return {
        "estimate":     estimate,
        "se":           se,
        "tost_ci_low":  tost_low,
        "tost_ci_high": tost_high,
        "margin":       margin,
        "equivalent":   equivalent,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 2. Covariate balance table
# ---------------------------------------------------------------------------

def covariate_balance_table(df, covariates, group_col, group_a_val, group_b_val,
                             label_a, label_b):
    """
    For each covariate, report mean/proportion and SD by subgroup plus the
    standardized mean difference (SMD): (mean_a - mean_b) / pooled_sd.

    Computed on the analysis-ready (post-imputation) covariate, matching
    what is actually fed into the model and the case-mix waterfall.
    |SMD| > 0.1 is the conventional threshold for "meaningful imbalance".
    """
    mask_a = df[group_col] == group_a_val
    mask_b = df[group_col] == group_b_val

    rows = []
    for col in covariates:
        a = pd.to_numeric(df.loc[mask_a, col], errors="coerce")
        b = pd.to_numeric(df.loc[mask_b, col], errors="coerce")
        n_a, n_b = int(a.notna().sum()), int(b.notna().sum())
        mean_a, mean_b = float(a.mean()), float(b.mean())
        sd_a, sd_b = float(a.std()), float(b.std())
        pooled_sd = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2) if (sd_a > 0 or sd_b > 0) else np.nan
        smd = (mean_a - mean_b) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

        rows.append({
            "covariate":   col,
            f"mean_{label_a}": mean_a, f"sd_{label_a}": sd_a, f"n_{label_a}": n_a,
            f"mean_{label_b}": mean_b, f"sd_{label_b}": sd_b, f"n_{label_b}": n_b,
            "smd":         smd,
            "meaningful_imbalance": (abs(smd) > 0.1) if not np.isnan(smd) else False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Cluster-bootstrap CI (e.g. resampling by patient rather than by row)
# ---------------------------------------------------------------------------

def cluster_bootstrap_gap(y_true, pred, g_test, cluster_ids, group_a_val, group_b_val,
                           metric_keys=("sensitivity", "fnr", "ppv", "npv",
                                        "predicted_positive_rate"),
                           n_boot=1000, seed=42):
    """
    Bootstrap CI for fairness gaps that resamples whole CLUSTERS (e.g.
    patients, each of whom may contribute multiple rows/encounters) rather
    than independent rows. Compares to the naive row-level bootstrap to
    test whether ignoring within-patient correlation understates the true
    uncertainty.

    Returns a DataFrame: metric, point, cluster_ci_low, cluster_ci_high,
    naive_ci_low, naive_ci_high, cluster_ci_width, naive_ci_width,
    ci_width_ratio_cluster_over_naive, n_clusters, n_rows.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred   = np.asarray(pred,   dtype=int)
    g_test = np.asarray(g_test)
    cluster_ids = np.asarray(cluster_ids)
    rng = np.random.default_rng(seed)

    mask_a = g_test == group_a_val
    mask_b = g_test == group_b_val

    def _gaps(idx):
        ma = fm.threshold_metrics(y_true[idx][mask_a[idx]], pred[idx][mask_a[idx]])
        mb = fm.threshold_metrics(y_true[idx][mask_b[idx]], pred[idx][mask_b[idx]])
        return {k: ma[k] - mb[k] for k in metric_keys}

    all_idx = np.arange(len(y_true))
    point = _gaps(all_idx)

    # cluster-level bootstrap: resample unique cluster ids with replacement,
    # take ALL rows belonging to sampled clusters (with repeats)
    unique_clusters = np.unique(cluster_ids)
    cluster_to_rows = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    n_clusters = len(unique_clusters)

    cluster_boot = {k: np.empty(n_boot) for k in metric_keys}
    naive_boot = {k: np.empty(n_boot) for k in metric_keys}

    for i in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_rows[c] for c in sampled_clusters])
        g = _gaps(idx)
        for k in metric_keys:
            cluster_boot[k][i] = g[k]

        idx_naive = rng.choice(all_idx, size=len(all_idx), replace=True)
        g_naive = _gaps(idx_naive)
        for k in metric_keys:
            naive_boot[k][i] = g_naive[k]

    rows = []
    for k in metric_keys:
        c_lo, c_hi = np.nanquantile(cluster_boot[k], [0.025, 0.975])
        n_lo, n_hi = np.nanquantile(naive_boot[k], [0.025, 0.975])
        rows.append({
            "metric":          k,
            "point":           point[k],
            "cluster_ci_low":  float(c_lo), "cluster_ci_high": float(c_hi),
            "naive_ci_low":    float(n_lo), "naive_ci_high":   float(n_hi),
            "cluster_ci_width": float(c_hi - c_lo),
            "naive_ci_width":   float(n_hi - n_lo),
            "ci_width_ratio_cluster_over_naive": float((c_hi - c_lo) / (n_hi - n_lo))
                                                   if (n_hi - n_lo) != 0 else np.nan,
            "n_clusters":      n_clusters,
            "n_rows":          len(y_true),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11. Prevalence-sweep curves
# ---------------------------------------------------------------------------

def prevalence_sweep_curve(metrics_a, metrics_b, prev_range=None, n_steps=50):
    """
    Sweep a common target prevalence across `prev_range` and recompute the
    Bayes-rule-adjusted PPV, NPV, and predicted-positive-rate gap at each
    value, holding each subgroup's own sensitivity/specificity fixed.

    Shows whether PPV/NPV/PPR gaps are stable properties of the classifier
    or depend on the prevalence context (they should depend on it; only a
    pure residual-error effect should be prevalence-invariant).

    Returns a DataFrame: target_prevalence, ppv_gap, npv_gap, ppr_gap,
    disparate_impact_ratio.
    """
    if prev_range is None:
        prev_range = (0.01, 0.50)
    grid = np.linspace(prev_range[0], prev_range[1], n_steps)

    sens_a, spec_a = metrics_a["sensitivity"], metrics_a["specificity"]
    sens_b, spec_b = metrics_b["sensitivity"], metrics_b["specificity"]

    rows = []
    for prev in grid:
        adj_a = ppv_npv_at_prevalence(sens_a, spec_a, prev)
        adj_b = ppv_npv_at_prevalence(sens_b, spec_b, prev)
        ppr_a, ppr_b = adj_a["predicted_positive_rate"], adj_b["predicted_positive_rate"]
        rows.append({
            "target_prevalence":      float(prev),
            "ppv_a":                  adj_a["ppv"], "ppv_b": adj_b["ppv"],
            "ppv_gap":                adj_a["ppv"] - adj_b["ppv"],
            "npv_a":                  adj_a["npv"], "npv_b": adj_b["npv"],
            "npv_gap":                adj_a["npv"] - adj_b["npv"],
            "ppr_a":                  ppr_a, "ppr_b": ppr_b,
            "ppr_gap":                ppr_a - ppr_b,
            "disparate_impact_ratio": (ppr_a / ppr_b) if ppr_b else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12. Dedicated prevalence-only synthetic positive control
#     (multi-seed, with explicit pass/fail recovery check)
# ---------------------------------------------------------------------------

def prevalence_only_control_replicated(n_seeds=20, n=20000,
                                        prev_a=0.25, prev_b=0.10,
                                        sens=0.80, spec=0.85):
    """
    Repeated runs of the pure prevalence-only synthetic control: group A
    and B share IDENTICAL sensitivity and specificity; only prevalence
    differs. Confirms the decomposition pipeline correctly attributes the
    resulting PPV/NPV gap to prevalence (not residual error).

    Returns (per_seed_df, summary_dict). The inherited prevalence-only pass
    criterion is |adjusted PPV gap| < 0.03 and |sensitivity gap| < 0.05.
    """
    rows = []
    for seed in range(n_seeds):
        r = synthetic_control(mechanism="prevalence_only", n=n, seed=seed)
        # override ground-truth params if caller asked for non-default values
        rows.append(r)
    df = pd.DataFrame(rows)

    n_recovered = int(df["decomposition_correctly_identifies"].sum())
    summary = {
        "n_seeds":               n_seeds,
        "n_recovered":           n_recovered,
        "recovery_rate":         n_recovered / n_seeds,
        "mean_ppv_gap_raw":      float(df["ppv_gap_raw"].mean()),
        "mean_ppv_gap_adj":      float(df["ppv_gap_adj"].mean()),
        "mean_ppv_attenuation_pct": float(df["ppv_attenuation"].mean()),
        "mean_sens_gap_raw":     float(df["sens_gap_raw"].mean()),
        "max_abs_sens_gap_raw":  float(df["sens_gap_raw"].abs().max()),
    }
    return df, summary


# ---------------------------------------------------------------------------
# 27. Covariate-standardized error gaps (g-computation + IPW)
# ---------------------------------------------------------------------------

def covariate_standardized_error_gap(y_true, pred, g_test, X_test,
                                      group_a_val, group_b_val,
                                      error_type="fn", seed=42):
    """
    Estimate what the FN-rate gap (among true positives) or FP-rate gap
    (among true negatives) would be if both subgroups had the SAME
    covariate distribution, using two methods:

      g-computation: fit ErrorOutcome ~ Sex + Covariates on the restricted
        population (positives for FN, negatives for FP); predict each
        individual's error probability twice -- once "as if group A",
        once "as if group B" -- and average over the SAME individuals.
        This standardizes by construction (every individual contributes
        to both counterfactual means).

      IPW: fit Sex ~ Covariates (propensity model) on the same restricted
        population; reweight each group's observed error rate by the
        inverse probability of its OWN sex given covariates, balancing the
        covariate distribution between groups.

    Returns dict: raw_gap, g_computation_gap, ipw_gap, attenuation_pct
    (using g-computation as the primary standardized estimate), plus the
    component pieces for transparency.
    """
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler

    y_true = np.asarray(y_true, dtype=int)
    pred   = np.asarray(pred,   dtype=int)
    g_test = np.asarray(g_test)
    X_test = np.asarray(X_test, dtype=float)

    if error_type == "fn":
        restrict = y_true == 1
        error_outcome_all = (pred == 0).astype(int)
    else:
        restrict = y_true == 0
        error_outcome_all = (pred == 1).astype(int)

    g_sub = g_test[restrict]
    mask_a, mask_b = g_sub == group_a_val, g_sub == group_b_val
    keep = mask_a | mask_b
    error_outcome = error_outcome_all[restrict][keep]
    X_sub = X_test[restrict][keep]
    sex_binary = mask_a[keep].astype(int)  # 1 = group A

    n_a, n_b = int(sex_binary.sum()), int((1 - sex_binary).sum())
    if n_a < 5 or n_b < 5:
        return {"raw_gap": np.nan, "g_computation_gap": np.nan, "ipw_gap": np.nan,
                "attenuation_pct": np.nan, "n_a": n_a, "n_b": n_b,
                "note": "insufficient sample size in one subgroup"}

    raw_gap = float(error_outcome[sex_binary == 1].mean() - error_outcome[sex_binary == 0].mean())

    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)
    design = np.column_stack([sex_binary, X_sub_s])

    # --- g-computation -----------------------------------------------------
    try:
        outcome_model = LR(max_iter=2000)
        outcome_model.fit(design, error_outcome)
        design_as_a = design.copy(); design_as_a[:, 0] = 1
        design_as_b = design.copy(); design_as_b[:, 0] = 0
        pred_as_a = outcome_model.predict_proba(design_as_a)[:, 1]
        pred_as_b = outcome_model.predict_proba(design_as_b)[:, 1]
        g_comp_gap = float(pred_as_a.mean() - pred_as_b.mean())
    except Exception:
        g_comp_gap = np.nan

    # --- IPW (propensity-weighted) -----------------------------------------
    try:
        propensity_model = LR(max_iter=2000)
        propensity_model.fit(X_sub_s, sex_binary)
        p_a = np.clip(propensity_model.predict_proba(X_sub_s)[:, 1], 0.01, 0.99)
        # weight A-rows by 1/p(A|X), B-rows by 1/(1-p(A|X)) -- standard
        # ATE-style inverse-probability weights, stabilized by the marginal
        # group share
        w = np.where(sex_binary == 1, sex_binary.mean() / p_a,
                     (1 - sex_binary.mean()) / (1 - p_a))
        w_a = w[sex_binary == 1]
        w_b = w[sex_binary == 0]
        ipw_mean_a = np.average(error_outcome[sex_binary == 1], weights=w_a)
        ipw_mean_b = np.average(error_outcome[sex_binary == 0], weights=w_b)
        ipw_gap = float(ipw_mean_a - ipw_mean_b)
    except Exception:
        ipw_gap = np.nan

    return {
        "raw_gap":            raw_gap,
        "g_computation_gap":  g_comp_gap,
        "ipw_gap":            ipw_gap,
        "attenuation_pct":    attenuation_pct(raw_gap, g_comp_gap),
        "n_a":                n_a, "n_b": n_b,
        "note":               "",
    }


# ---------------------------------------------------------------------------
# 28. Risk-decile stratified error analysis
# ---------------------------------------------------------------------------

def risk_decile_error_table(y_true, prob, pred, g_test, group_a_val, group_b_val,
                             label_a, label_b, n_deciles=10):
    """
    Split the test set into predicted-risk deciles (decile 1 = lowest
    scores) and report, within each decile and subgroup: n, n_pos, n_neg,
    observed prevalence, sensitivity, FNR, specificity, FPR, PPV, NPV, and
    mean predicted vs. observed event rate (a simple within-decile
    calibration check).

    Shows WHERE a subgroup gap is concentrated -- low-risk deciles (the
    model systematically under-scores one subgroup), the threshold region,
    or high-risk deciles (severe cases handled differently).
    """
    y_true = np.asarray(y_true, dtype=int)
    prob   = np.asarray(prob,   dtype=float)
    pred   = np.asarray(pred,   dtype=int)
    g_test = np.asarray(g_test)

    try:
        deciles = pd.qcut(prob, n_deciles, labels=False, duplicates="drop")
    except Exception:
        deciles = pd.cut(prob, n_deciles, labels=False)

    rows = []
    for d in sorted(pd.unique(deciles[~pd.isna(deciles)])):
        in_decile = deciles == d
        for val, lbl in [(group_a_val, label_a), (group_b_val, label_b)]:
            mask = in_decile & (g_test == val)
            n = int(mask.sum())
            if n == 0:
                continue
            m = fm.threshold_metrics(y_true[mask], pred[mask])
            rows.append({
                "decile":            int(d) + 1,
                "score_range_low":   float(prob[mask].min()),
                "score_range_high":  float(prob[mask].max()),
                "group":             lbl,
                "n":                 n,
                "n_pos":             m["n_pos"], "n_neg": m["n_neg"],
                "observed_prevalence": m["prevalence"],
                "mean_predicted_score": float(prob[mask].mean()),
                "sensitivity":       m["sensitivity"], "fnr": m["fnr"],
                "specificity":       m["specificity"], "fpr": m["fpr"],
                "ppv":               m["ppv"], "npv": m["npv"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 29. Subgroup-by-covariate interaction tests
# ---------------------------------------------------------------------------

def subgroup_covariate_interaction_test(y_true, pred, g_test, moderator_values,
                                         group_a_val, group_b_val,
                                         adjustment_cols=None, X_test=None,
                                         error_type="fn"):
    """
    Tests whether the sex effect on model error is uniform or concentrated
    in a clinical stratum, via:

        ErrorOutcome ~ Sex + Moderator + Sex*Moderator [+ adjustment covariates]

    restricted to true positives (error_type="fn") or true negatives
    (error_type="fp"). `moderator_values` is a 1-D array aligned to the
    full (unrestricted) y_true/pred/g_test; it is standardized before
    fitting so the interaction coefficient is on a comparable scale across
    moderators.

    Every result has an explicit ``status`` (``estimable`` or
    ``nonestimable``), ``warning``, and ``error``. On a successful fit,
    returns: interaction_logOR, interaction_OR,
    interaction_OR_ci_low, interaction_OR_ci_high, interaction_p,
    main_effect_sex_OR, main_effect_moderator_OR, n_obs, n_error, note.
    Insufficient-sample, failed, unconverged, warned, or nonfinite fits are
    retained as ``status=nonestimable`` with all inferential values missing;
    warnings/errors are never converted into numeric results.
    """
    from sklearn.preprocessing import StandardScaler

    y_true = np.asarray(y_true, dtype=int)
    pred   = np.asarray(pred,   dtype=int)
    g_test = np.asarray(g_test)
    moderator_values = np.asarray(moderator_values, dtype=float)

    if error_type == "fn":
        restrict = y_true == 1
        error_outcome_all = (pred == 0).astype(int)
    else:
        restrict = y_true == 0
        error_outcome_all = (pred == 1).astype(int)

    g_sub = g_test[restrict]
    mask_a, mask_b = g_sub == group_a_val, g_sub == group_b_val
    keep = mask_a | mask_b

    error_outcome = error_outcome_all[restrict][keep]
    sex_binary = mask_a[keep].astype(float)
    mod_sub = moderator_values[restrict][keep]

    valid = ~np.isnan(mod_sub)
    error_outcome, sex_binary, mod_sub = error_outcome[valid], sex_binary[valid], mod_sub[valid]

    inferential_fields = (
        "interaction_logOR", "interaction_OR", "interaction_OR_ci_low",
        "interaction_OR_ci_high", "interaction_p", "main_effect_sex_OR",
        "main_effect_moderator_OR",
    )

    def nonestimable(note, *, warning="", error=""):
        return {
            "status": "nonestimable",
            **{field: np.nan for field in inferential_fields},
            "n_obs": len(error_outcome),
            "n_error": int(error_outcome.sum()),
            "warning": warning,
            "error": error,
            "note": note,
        }

    if len(error_outcome) < 20 or error_outcome.sum() < 5 or sex_binary.sum() < 5:
        return nonestimable("insufficient sample size")

    scaler = StandardScaler()
    mod_s = scaler.fit_transform(mod_sub.reshape(-1, 1)).ravel()
    interaction = sex_binary * mod_s

    design_cols = {"sex": sex_binary, "moderator": mod_s, "sex_x_moderator": interaction}
    if adjustment_cols is not None and X_test is not None:
        X_sub = np.asarray(X_test, dtype=float)[restrict][keep][valid]
        adj_scaled = StandardScaler().fit_transform(X_sub[:, adjustment_cols])
        for j in range(adj_scaled.shape[1]):
            design_cols[f"adj_{j}"] = adj_scaled[:, j]

    design = pd.DataFrame(design_cols)

    try:
        exog = sm.add_constant(design, has_constant="add")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.Logit(error_outcome, exog).fit(
                disp=0, method="bfgs", maxiter=200
            )
        coef = float(fit.params.get("sex_x_moderator", np.nan))
        ci = fit.conf_int()
        ci_lo = float(ci.loc["sex_x_moderator", 0]) if "sex_x_moderator" in ci.index else np.nan
        ci_hi = float(ci.loc["sex_x_moderator", 1]) if "sex_x_moderator" in ci.index else np.nan
        p = float(fit.pvalues.get("sex_x_moderator", np.nan))
        with np.errstate(over="ignore", invalid="ignore"):
            interaction_or = float(np.exp(coef))
            ci_or_low = float(np.exp(ci_lo))
            ci_or_high = float(np.exp(ci_hi))
            sex_or = float(np.exp(fit.params.get("sex", np.nan)))
            mod_or = float(np.exp(fit.params.get("moderator", np.nan)))
    except Exception as e:
        return nonestimable(
            "fit failed", error=f"{type(e).__name__}: {e}"
        )

    warning_text = " | ".join(
        f"{item.category.__name__}: {item.message}" for item in caught
    )
    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
    estimates = (coef, interaction_or, ci_or_low, ci_or_high, p, sex_or, mod_or)
    if warning_text or not converged or not all(np.isfinite(value) for value in estimates):
        reasons = []
        if warning_text:
            reasons.append("fit emitted warning")
        if not converged:
            reasons.append("optimizer did not converge")
        if not all(np.isfinite(value) for value in estimates):
            reasons.append("inferential values were nonfinite")
        return nonestimable(
            "; ".join(reasons), warning=warning_text
        )

    return {
        "status":             "estimable",
        "interaction_logOR": coef,
        "interaction_OR":    interaction_or,
        "interaction_OR_ci_low":  ci_or_low,
        "interaction_OR_ci_high": ci_or_high,
        "interaction_p":     p,
        "main_effect_sex_OR": sex_or,
        "main_effect_moderator_OR": mod_or,
        "n_obs":             len(error_outcome),
        "n_error":           int(error_outcome.sum()),
        "warning":           "",
        "error":             "",
        "note":              "",
    }


# ---------------------------------------------------------------------------
# 37. Balanced-training analysis
# ---------------------------------------------------------------------------

def balanced_training_strategies(X_train, y_train, g_train, group_a_val, group_b_val,
                                  random_state=42):
    """
    Build resampled/reweighted versions of the training set under several
    balancing schemes (a base learner is fit by the caller on each):

      original                    -- no resampling, no weights
      equal_sample_size            -- undersample the larger sex group to
                                       match the smaller group's N
      equal_positive_cases          -- undersample positives in whichever
                                       sex group has more, so both groups
                                       contribute the same number of cases
      equal_prevalence_by_sex       -- downsample within each sex group so
                                       both groups share the SAME outcome
                                       prevalence (the smaller of the two)
      class_balanced_weighting      -- sample_weight inversely proportional
                                       to the POOLED outcome class frequency
      subgroup_balanced_weighting   -- sample_weight inversely proportional
                                       to sex-group frequency (each sex
                                       contributes equally regardless of N)
      positive_case_subgroup_reweighting -- sample_weight upweights positive
                                       cases from whichever sex has fewer
                                       positives, balancing case
                                       contribution across sexes

    Returns {scheme_name: (X_resampled, y_resampled, sample_weight_or_None)}.
    Resampling schemes return weight=None (already resampled); reweighting
    schemes return the original X/y plus a sample_weight array.
    """
    rng = np.random.default_rng(random_state)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    g_train = np.asarray(g_train)

    mask_a, mask_b = g_train == group_a_val, g_train == group_b_val
    idx_a, idx_b = np.where(mask_a)[0], np.where(mask_b)[0]

    out = {"original": (X_train, y_train, None)}

    # equal sample size
    n_min = min(len(idx_a), len(idx_b))
    sel_a = rng.choice(idx_a, size=n_min, replace=False)
    sel_b = rng.choice(idx_b, size=n_min, replace=False)
    sel = np.concatenate([sel_a, sel_b])
    out["equal_sample_size"] = (X_train[sel], y_train[sel], None)

    # equal positive cases (downsample positives in the larger-positive group)
    pos_a, pos_b = idx_a[y_train[idx_a] == 1], idx_b[y_train[idx_b] == 1]
    neg_a, neg_b = idx_a[y_train[idx_a] == 0], idx_b[y_train[idx_b] == 0]
    n_pos_min = min(len(pos_a), len(pos_b))
    sel_pos_a = rng.choice(pos_a, size=n_pos_min, replace=False)
    sel_pos_b = rng.choice(pos_b, size=n_pos_min, replace=False)
    sel = np.concatenate([sel_pos_a, sel_pos_b, neg_a, neg_b])
    out["equal_positive_cases"] = (X_train[sel], y_train[sel], None)

    # equal prevalence by sex (match the lower of the two group prevalences,
    # by downsampling positives in whichever group has the higher prevalence)
    prev_a = len(pos_a) / len(idx_a) if len(idx_a) else np.nan
    prev_b = len(pos_b) / len(idx_b) if len(idx_b) else np.nan
    target_prev = min(prev_a, prev_b)
    n_pos_a_target = int(round(target_prev * len(idx_a)))
    n_pos_b_target = int(round(target_prev * len(idx_b)))
    sel_pos_a2 = rng.choice(pos_a, size=min(n_pos_a_target, len(pos_a)), replace=False)
    sel_pos_b2 = rng.choice(pos_b, size=min(n_pos_b_target, len(pos_b)), replace=False)
    sel = np.concatenate([sel_pos_a2, sel_pos_b2, neg_a, neg_b])
    out["equal_prevalence_by_sex"] = (X_train[sel], y_train[sel], None)

    # class-balanced weighting (pooled outcome class)
    n_pos_total, n_neg_total = int((y_train == 1).sum()), int((y_train == 0).sum())
    w_class = np.where(y_train == 1, len(y_train) / (2 * n_pos_total),
                        len(y_train) / (2 * n_neg_total))
    out["class_balanced_weighting"] = (X_train, y_train, w_class)

    # subgroup-balanced weighting (each sex contributes equally)
    w_subgroup = np.where(mask_a, len(y_train) / (2 * len(idx_a)),
                           len(y_train) / (2 * len(idx_b)))
    out["subgroup_balanced_weighting"] = (X_train, y_train, w_subgroup)

    # positive-case subgroup reweighting (upweight positives in the
    # sex group with fewer positives; negatives weighted 1.0)
    w_pos_reweight = np.ones(len(y_train))
    n_pos_a, n_pos_b = len(pos_a), len(pos_b)
    if n_pos_a > 0 and n_pos_b > 0:
        target_n = max(n_pos_a, n_pos_b)
        w_pos_reweight[pos_a] = target_n / n_pos_a
        w_pos_reweight[pos_b] = target_n / n_pos_b
    out["positive_case_subgroup_reweighting"] = (X_train, y_train, w_pos_reweight)

    return out


# ---------------------------------------------------------------------------
# 38. Additional protected attributes / intersectional groups
#     (thin helpers that recompute gaps on an ALREADY-FITTED model's
#     test-set predictions, using a different grouping variable -- this
#     audits the actual model, not a separately retrained one)
# ---------------------------------------------------------------------------

def additional_attribute_gap_row(y_test, prob_te, attr_values, val_a, val_b,
                                  threshold, extra=None):
    """
    Recompute the full gap row (AUROC/AUPRC/calibration + raw and
    prevalence-adjusted PPV/NPV gaps, sensitivity/FNR/specificity/FPR/PPR
    gaps) for a NEW grouping variable (e.g. race, age group, income),
    reusing the SAME fitted model's predictions. Rows where attr_values is
    not in {val_a, val_b} are excluded.
    """
    y_test = np.asarray(y_test, dtype=int)
    prob_te = np.asarray(prob_te, dtype=float)
    attr_values = np.asarray(attr_values)

    keep = np.isin(attr_values, [val_a, val_b])
    y_sub, prob_sub, attr_sub = y_test[keep], prob_te[keep], attr_values[keep]
    mask_a_sub = attr_sub == val_a
    mask_b_sub = attr_sub == val_b

    if mask_a_sub.sum() < 10 or mask_b_sub.sum() < 10:
        row = {"note": "insufficient sample size", "n_a": int(mask_a_sub.sum()),
               "n_b": int(mask_b_sub.sum())}
        if extra:
            row.update(extra)
        return row

    row = _gap_row(y_sub, prob_sub, mask_a_sub, mask_b_sub, threshold=threshold, extra=extra)
    row["n_a"] = int(mask_a_sub.sum())
    row["n_b"] = int(mask_b_sub.sum())
    row["note"] = ""
    return row


def stratified_sex_gap(y_test, prob_te, g_test, group_a_val, group_b_val,
                        strat_values, strat_level, threshold, extra=None):
    """
    Compute the sex gap (group_a_val vs group_b_val) restricted to rows
    where `strat_values == strat_level` -- the intersectional check of
    whether the sex-fairness pattern is uniform across another attribute
    (e.g. race, income) or concentrated in one stratum.
    """
    y_test = np.asarray(y_test, dtype=int)
    prob_te = np.asarray(prob_te, dtype=float)
    g_test = np.asarray(g_test)
    strat_values = np.asarray(strat_values)

    keep = strat_values == strat_level
    y_sub, prob_sub, g_sub = y_test[keep], prob_te[keep], g_test[keep]
    mask_a_sub = g_sub == group_a_val
    mask_b_sub = g_sub == group_b_val

    if mask_a_sub.sum() < 10 or mask_b_sub.sum() < 10:
        row = {"note": "insufficient sample size", "n_a": int(mask_a_sub.sum()),
               "n_b": int(mask_b_sub.sum())}
        if extra:
            row.update(extra)
        return row

    row = _gap_row(y_sub, prob_sub, mask_a_sub, mask_b_sub, threshold=threshold, extra=extra)
    row["n_a"] = int(mask_a_sub.sum())
    row["n_b"] = int(mask_b_sub.sum())
    row["note"] = ""
    return row
