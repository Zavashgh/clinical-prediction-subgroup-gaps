"""
metrics.py
==========
Per-subgroup performance, calibration, and fairness-gap metrics.

All functions operate on 1-D numpy arrays:
  y_true : ground-truth binary labels (0/1)
  y_prob : predicted probability of the positive class, in [0, 1]
  y_pred : thresholded binary predictions (0/1)

Definitions follow standard epidemiology / ML-fairness usage:
  - sensitivity (recall, TPR) = TP / (TP + FN)
  - specificity              = TN / (TN + FP)
  - FPR = 1 - specificity, FNR = 1 - sensitivity
  - PPV (precision)          = TP / (TP + FP)
  - NPV                       = TN / (TN + FN)
  - prevalence                = (TP + FN) / N   (base rate of the outcome)
  - predicted_positive_rate   = (TP + FP) / N   (a.k.a. "selection rate")

Note: sensitivity and specificity are properties of the classifier's
operating point that do *not* depend on prevalence. PPV, NPV, and the
predicted-positive rate *do* depend on prevalence -- this is exactly the
fact exploited by the prevalence adjustment in adjustments.py.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def wilson_binomial_ci(successes, trials, confidence=0.95):
    """Wilson score interval for a binomial proportion.

    Returns missing bounds when the metric denominator is zero. The default
    95% interval uses the standard-normal 0.975 quantile.
    """
    from statistics import NormalDist

    successes, trials = int(successes), int(trials)
    if trials <= 0:
        return {"ci_low": np.nan, "ci_high": np.nan, "method": "wilson"}
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return {
        "ci_low": max(0.0, centre - radius),
        "ci_high": min(1.0, centre + radius),
        "method": "wilson",
    }


def threshold_metric_confidence_intervals(metrics, confidence=0.95):
    """Wilson intervals for reportable group-specific threshold metrics."""
    specifications = {
        "prevalence": (metrics["n_pos"], metrics["n"]),
        "sensitivity": (metrics["tp"], metrics["n_pos"]),
        "specificity": (metrics["tn"], metrics["n_neg"]),
        "ppv": (metrics["tp"], metrics["tp"] + metrics["fp"]),
        "npv": (metrics["tn"], metrics["tn"] + metrics["fn"]),
    }
    return {
        name: {
            "point": metrics[name],
            **wilson_binomial_ci(successes, trials, confidence),
            "numerator": int(successes),
            "denominator": int(trials),
        }
        for name, (successes, trials) in specifications.items()
    }


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, tn, fn) as plain ints."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def threshold_metrics(y_true, y_pred):
    """Confusion-matrix-derived metrics at a fixed decision threshold."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    n = tp + fp + tn + fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    fpr = 1 - specificity if not np.isnan(specificity) else np.nan
    fnr = 1 - sensitivity if not np.isnan(sensitivity) else np.nan

    if not np.isnan(ppv) and not np.isnan(sensitivity) and (ppv + sensitivity) > 0:
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
    else:
        f1 = np.nan

    return {
        "n": n,
        "n_pos": tp + fn,
        "n_neg": tn + fp,
        "prevalence": (tp + fn) / n if n > 0 else np.nan,
        "predicted_positive_rate": (tp + fp) / n if n > 0 else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def discrimination_metrics(y_true, y_prob):
    """AUROC and AUPRC. Returns NaN if a subgroup has only one class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    }


# Weak-calibration (intercept/slope) numerical guards.
#
# The final models are isotonically calibrated, so predicted probabilities are a
# step function that can contain exact 0 and 1 and heavy ties. Clipping those to
# CALIBRATION_LOGIT_EPS maps them to logits near +/- 13.8, which are high
# leverage points for the recalibration fit.
#
# Reportability is decided by EXPLICIT NUMERICAL CRITERIA ONLY: convergence,
# finiteness, an interior optimum, and a non-degenerate predictor. There is
# deliberately no cutoff on the magnitude of the fitted coefficients. An earlier
# version rejected any |coefficient| above 25, which has no methodological
# basis: a large slope that the solver reached cleanly is a real estimate of a
# badly calibrated model, and silently converting it to NA would hide exactly
# the miscalibration the reader needs to see. Separation and non-convergence are
# detected directly instead, and reported by name.
CALIBRATION_LOGIT_EPS = 1e-6
CALIBRATION_MAX_ITER = 100
CALIBRATION_TOLERANCE = 1e-8
SEPARATION_PROBABILITY_TOLERANCE = 1e-10


def calibration_logit(y_prob):
    """Clip to the open unit interval and return the logit."""
    p = np.clip(np.asarray(y_prob, dtype=float),
                CALIBRATION_LOGIT_EPS, 1.0 - CALIBRATION_LOGIT_EPS)
    return np.log(p / (1.0 - p))


def fit_calibration_intercept_slope(y_true, logit_p):
    """Newton fit of ``y ~ a + b * logit_p``.

    Returns ``(intercept, slope, status)``. ``status`` is ``"ok"`` only when the
    solver converged to a finite interior optimum. Otherwise the coefficients
    are NaN and the status names the reason, one of ``single_class``,
    ``nonfinite_predictor``, ``degenerate_predictor``, ``separation``,
    ``singular_information_matrix``, ``nonfinite_step``, ``nonfinite_fit`` or
    ``not_converged``.

    A direct two-parameter Newton solve is used rather than a general logistic
    regression because the bootstrap needs thousands of these fits per
    comparison, and because it exposes the convergence state directly. A general
    solver can RETURN a non-converged result without raising, which is how a
    separated fit previously reached the output as though it were an estimate.
    """
    y = np.asarray(y_true, dtype=float)
    x = np.asarray(logit_p, dtype=float)

    if np.count_nonzero(y == 1.0) == 0 or np.count_nonzero(y == 0.0) == 0:
        return np.nan, np.nan, "single_class"
    if not np.isfinite(x).all():
        return np.nan, np.nan, "nonfinite_predictor"
    if np.ptp(x) == 0.0:
        return np.nan, np.nan, "degenerate_predictor"

    beta = np.zeros(2, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(CALIBRATION_MAX_ITER):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
        if not np.isfinite(mu).all():
            return np.nan, np.nan, "nonfinite_fit"
        weight = mu * (1.0 - mu)
        if np.max(weight) < SEPARATION_PROBABILITY_TOLERANCE:
            return np.nan, np.nan, "separation"
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, "singular_information_matrix"
        if not np.isfinite(step).all():
            return np.nan, np.nan, "nonfinite_step"
        beta = beta + step
        if np.max(np.abs(step)) < CALIBRATION_TOLERANCE:
            if not np.isfinite(beta).all():
                return np.nan, np.nan, "nonfinite_fit"
            return float(beta[0]), float(beta[1]), "ok"
    return np.nan, np.nan, "not_converged"


def calibration_metrics(y_true, y_prob, n_bins=10):
    """Brier score, expected calibration error, and calibration intercept/slope.

    ``calib_status`` records why an intercept/slope pair is NaN when it is, and
    ``calib_n_clipped`` reports how many probabilities sat at or beyond the
    clipping boundary, so the amount of clipping is visible rather than implicit.
    ECE is descriptive: it depends on the arbitrary bin count.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_prob)) * abs(
            y_prob[mask].mean() - y_true[mask].mean()
        )

    eps = CALIBRATION_LOGIT_EPS
    n_clipped = int(np.sum((y_prob <= eps) | (y_prob >= 1.0 - eps)))
    intercept, slope, status = fit_calibration_intercept_slope(
        y_true, calibration_logit(y_prob)
    )
    return {
        "brier": brier,
        "ece": ece,
        "calib_intercept": intercept,
        "calib_slope": slope,
        "calib_status": status,
        "calib_n_clipped": n_clipped,
    }


def subgroup_metrics(y_true, y_prob, y_pred, n_bins=10):
    """All per-subgroup metrics in one dict: threshold + discrimination + calibration."""
    out = {}
    out.update(threshold_metrics(y_true, y_pred))
    out.update(discrimination_metrics(y_true, y_prob))
    out.update(calibration_metrics(y_true, y_prob, n_bins=n_bins))
    return out


def fairness_gaps(metrics_a, metrics_b):
    """
    Gap = metric(group A) - metric(group B), for the metrics used to
    define common group-fairness criteria:

      - equal_opportunity_diff   : sensitivity (TPR) gap
      - equalized_odds_diff      : max(|TPR gap|, |FPR gap|)
      - demographic_parity_diff  : predicted-positive-rate gap
      - disparate_impact_ratio   : predicted-positive-rate(A) / predicted-positive-rate(B)
      - ppv_diff, npv_diff, fnr_diff, auroc_diff : additional diagnostic gaps

    `metrics_a` / `metrics_b` are dicts as returned by `subgroup_metrics`.
    """
    eod_components = []
    for k in ("sensitivity", "fpr"):
        a, b = metrics_a[k], metrics_b[k]
        if not (np.isnan(a) or np.isnan(b)):
            eod_components.append(abs(a - b))

    ppr_b = metrics_b["predicted_positive_rate"]
    disparate_impact = (
        metrics_a["predicted_positive_rate"] / ppr_b
        if ppr_b not in (0, None) and not np.isnan(ppr_b) and ppr_b != 0
        else np.nan
    )

    return {
        "equal_opportunity_diff": metrics_a["sensitivity"] - metrics_b["sensitivity"],
        "equalized_odds_diff": max(eod_components) if eod_components else np.nan,
        "demographic_parity_diff": metrics_a["predicted_positive_rate"] - metrics_b["predicted_positive_rate"],
        "disparate_impact_ratio": disparate_impact,
        "ppv_diff": metrics_a["ppv"] - metrics_b["ppv"],
        "npv_diff": metrics_a["npv"] - metrics_b["npv"],
        "fnr_diff": metrics_a["fnr"] - metrics_b["fnr"],
        "auroc_diff": metrics_a["auroc"] - metrics_b["auroc"],
    }
