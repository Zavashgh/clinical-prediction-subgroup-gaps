"""
adjustments.py
===============
The "is this gap real?" toolkit. Implements:

  1. Prevalence adjustment   -- recompute prevalence-sensitive metrics
                                 (PPV, NPV, predicted-positive rate) as if
                                 both subgroups had the same disease
                                 prevalence, holding sensitivity/specificity
                                 (the model's "true" operating characteristics)
                                 fixed.
  2. Calibration adjustment  -- fit a subgroup-specific recalibration map
                                 and compare a shared threshold vs.
                                 thresholds chosen to equalize sensitivity.
  3. Case-mix adjustment     -- sequentially add covariate blocks to a
                                 logistic regression and see how much each
                                 block explains of the group gap (waterfall).
  4. Bootstrap CIs           -- resample each subgroup with replacement to
                                 get confidence intervals on any gap, raw or
                                 adjusted.

Throughout, "group A" / "group B" follow the convention set by the caller
(e.g. A = Male, B = Female); a gap is always A - B.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from . import metrics as fm


# ---------------------------------------------------------------------------
# 1. Prevalence adjustment
# ---------------------------------------------------------------------------

def ppv_npv_at_prevalence(sensitivity, specificity, prevalence):
    """
    Recompute PPV, NPV, and predicted-positive rate at an arbitrary
    prevalence, holding sensitivity and specificity fixed.

    These are the standard Bayes'-rule relationships:
        PPV  = sens * prev / [ sens * prev + (1-spec) * (1-prev) ]
        NPV  = spec * (1-prev) / [ spec * (1-prev) + (1-sens) * prev ]
        PPR  = sens * prev + (1-spec) * (1-prev)   (= P(predicted positive))
    """
    sens, spec, prev = sensitivity, specificity, prevalence

    ppv_denom = sens * prev + (1 - spec) * (1 - prev)
    ppv = (sens * prev) / ppv_denom if ppv_denom > 0 else np.nan

    npv_denom = spec * (1 - prev) + (1 - sens) * prev
    npv = (spec * (1 - prev)) / npv_denom if npv_denom > 0 else np.nan

    ppr = sens * prev + (1 - spec) * (1 - prev)

    return {"ppv": ppv, "npv": npv, "predicted_positive_rate": ppr}


def prevalence_adjusted_metrics(metrics_a, metrics_b, target_prevalence=None):
    """
    Recompute PPV / NPV / predicted-positive-rate / disparate-impact gaps
    as if both groups shared `target_prevalence` (default: the pooled
    prevalence implied by both groups' n_pos / n).

    Returns a dict with raw gap, adjusted gap, and percent attenuation
    for each prevalence-sensitive metric.
    """
    if target_prevalence is None:
        n_pos = metrics_a["n_pos"] + metrics_b["n_pos"]
        n_tot = metrics_a["n"] + metrics_b["n"]
        target_prevalence = n_pos / n_tot

    adj_a = ppv_npv_at_prevalence(metrics_a["sensitivity"], metrics_a["specificity"], target_prevalence)
    adj_b = ppv_npv_at_prevalence(metrics_b["sensitivity"], metrics_b["specificity"], target_prevalence)

    out = {"target_prevalence": target_prevalence}
    for key in ("ppv", "npv", "predicted_positive_rate"):
        raw_gap = metrics_a[key] - metrics_b[key]
        adj_gap = adj_a[key] - adj_b[key]
        out[key] = {
            "raw_a": metrics_a[key],
            "raw_b": metrics_b[key],
            "raw_gap": raw_gap,
            "adjusted_a": adj_a[key],
            "adjusted_b": adj_b[key],
            "adjusted_gap": adj_gap,
            "attenuation_pct": attenuation_pct(raw_gap, adj_gap),
        }

    # disparate impact ratio, raw vs adjusted
    raw_di = (metrics_a["predicted_positive_rate"] / metrics_b["predicted_positive_rate"]
              if metrics_b["predicted_positive_rate"] else np.nan)
    adj_di = (adj_a["predicted_positive_rate"] / adj_b["predicted_positive_rate"]
              if adj_b["predicted_positive_rate"] else np.nan)
    out["disparate_impact_ratio"] = {"raw": raw_di, "adjusted": adj_di}

    return out


def attenuation_pct(raw_gap, adjusted_gap):
    """
    Signed percent change in the gap after adjustment.

    This is 100 * (raw_gap - adjusted_gap) / raw_gap, so:
    100%  -> adjusted gap is zero
    0%    -> adjusted gap is unchanged
    <0%   -> adjusted gap is farther from zero in the original direction
    >100% -> adjusted gap has reversed sign

    The ratio is unstable when the raw gap is close to zero and should not be
    interpreted in that setting. Raw and adjusted gaps must always be reported
    alongside it so the direction and magnitude remain visible.
    """
    if raw_gap == 0 or np.isnan(raw_gap) or np.isnan(adjusted_gap):
        return np.nan
    return 100.0 * (raw_gap - adjusted_gap) / raw_gap


# ---------------------------------------------------------------------------
# 2. Calibration adjustment
# ---------------------------------------------------------------------------

def fit_subgroup_calibrators(y_true, y_prob, group, groups, method="isotonic"):
    """
    Fit a separate recalibration map (isotonic regression on predicted
    probability) within each subgroup.

    Returns {group_value: fitted IsotonicRegression}.
    """
    calibrators = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    for g in groups:
        mask = group == g
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(y_prob[mask], y_true[mask])
        calibrators[g] = ir
    return calibrators


def equal_sensitivity_thresholds(y_true, y_prob, group, groups, target_sensitivity):
    """
    For each subgroup, find the threshold whose achieved sensitivity is the
    smallest attainable value greater than or equal to `target_sensitivity`.

    Exact equality may be impossible when positive cases have tied scores;
    all cases tied at the boundary are included by the downstream >= rule.
    Callers must state which data were used to derive these thresholds. If
    derivation and evaluation use the same final test outcomes, the result is
    exploratory/descriptive rather than a validated threshold policy.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    if (not np.isfinite(target_sensitivity)
            or not 0.0 <= target_sensitivity <= 1.0):
        raise ValueError("target_sensitivity must be a finite value in [0, 1]")

    thresholds = {}
    for g in groups:
        mask = group == g
        yt, yp = y_true[mask], y_prob[mask]

        if len(yt) == 0:
            raise ValueError(f"Group {g!r} has no observations")

        pos_probs = np.sort(yp[yt == 1])[::-1]
        if len(pos_probs) == 0:
            raise ValueError(f"Group {g!r} has no positive observations")

        n_keep = int(np.ceil(target_sensitivity * len(pos_probs)))
        if n_keep == 0:
            # Put the threshold just above every observed score so no case is
            # classified positive and sensitivity is exactly zero.
            thresholds[g] = float(np.nextafter(np.max(yp), np.inf))
        else:
            # With the downstream >= comparison, n_keep positives correspond
            # to zero-based boundary index n_keep - 1 (absent boundary ties).
            thresholds[g] = float(pos_probs[n_keep - 1])
    return thresholds


# ---------------------------------------------------------------------------
# 3. Case-mix adjustment (waterfall)
# ---------------------------------------------------------------------------

def case_mix_waterfall(df, target_col, group_col, group_a, group_b,
                        covariate_blocks, random_state=42):
    """
    Sequentially add blocks of covariates to a logistic regression that
    predicts the OUTCOME, and track the group coefficient (the log-odds
    gap between group A and B, holding the included covariates fixed).

    This answers: "how much of the raw outcome gap between groups is
    explained by each block of covariates (demographics, comorbidities,
    etc.)?" It is a case-mix decomposition, not a model-fairness metric --
    it operates on the *outcome*, independent of any predictive model.

    Returns a DataFrame with one row per step:
      step, block_added, group_coef, group_coef_pvalue, pct_of_raw_explained
    """
    work = df[df[group_col].isin([group_a, group_b])].copy()
    work["_group_a"] = (work[group_col] == group_a).astype(int)

    rows = []
    cumulative_features = ["_group_a"]
    raw_coef = None

    step_names = ["raw (group only)"] + list(covariate_blocks.keys())
    for step_name in step_names:
        if step_name != "raw (group only)":
            cumulative_features += [c for c in covariate_blocks[step_name] if c in work.columns]

        X = work[cumulative_features].astype(float)
        y = work[target_col].astype(int)

        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)
        group_coef = model.coef_[0][cumulative_features.index("_group_a")]

        if raw_coef is None:
            raw_coef = group_coef

        pct_explained = (
            100.0 * (raw_coef - group_coef) / raw_coef if raw_coef != 0 else np.nan
        )
        rows.append({
            "step": step_name,
            "n_features": len(cumulative_features),
            "group_coef_logodds": group_coef,
            "pct_of_raw_gap_explained": pct_explained,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Bootstrap confidence intervals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Superseded clustered resampling
# ---------------------------------------------------------------------------
#
# The helpers below drew a SEPARATE patient pool for each subgroup, so a patient
# whose encounters carry more than one demographic label received two
# independent bootstrap multiplicities. That breaks the within-patient
# dependence the clustering exists to preserve, and it is the design an
# independent audit identified as incorrect.
#
# The correct implementation is uncertainty.bootstrap_comparison, which draws
# each patient ONCE per replicate across the whole held-out comparison and
# applies that single multiplicity to all of the patient's encounters in either
# subgroup.
#
# These entry points are kept so that code or notebooks referring to them fail
# loudly instead of silently producing the superseded intervals.

CLUSTERED_BOOTSTRAP_REPLACEMENT = (
    "Use src.uncertainty.bootstrap_comparison, which draws each patient once "
    "per replicate across the whole held-out comparison and applies that "
    "multiplicity to all of the patient's encounters regardless of subgroup. "
    "The subgroup-separate patient bootstrap that used to live here is "
    "superseded: it gave a patient appearing under two demographic labels two "
    "independent multiplicities."
)


class SupersededResamplingDesign(RuntimeError):
    """Raised when superseded subgroup-separate patient resampling is requested."""


def _refuse_clustered(function_name):
    raise SupersededResamplingDesign(
        f"{function_name} implemented subgroup-separate patient resampling, "
        f"which is superseded. {CLUSTERED_BOOTSTRAP_REPLACEMENT}"
    )


def bootstrap_gap_ci_clustered(*args, **kwargs):
    """Superseded. Raises; see CLUSTERED_BOOTSTRAP_REPLACEMENT."""
    _refuse_clustered("bootstrap_gap_ci_clustered")


def bootstrap_gap_ci(y_true, y_prob, y_pred, mask_a, mask_b, metric_key,
                      n_boot=1000, seed=42, ci=0.95,
                      prevalence_adjust=False, target_prevalence=None):
    """
    Bootstrap CI for the gap (group A - group B) of a single metric.

    Each bootstrap iteration resamples group A and group B *separately*
    (with replacement, same size as the original subgroup), recomputes
    the metric for each, and takes the difference. This preserves each
    subgroup's sample size and prevalence in every resample.

    If `prevalence_adjust=True`, `metric_key` must be one of
    {"ppv", "npv", "predicted_positive_rate"}, and each resample's gap is
    computed *after* standardizing both groups to `target_prevalence`
    (default: pooled prevalence of that resample).

    Returns {"point": ..., "ci_low": ..., "ci_high": ..., "boot": array}
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]

    def _metric(idx):
        # Only threshold-derived counts are needed for the gaps bootstrapped
        # here (sensitivity, specificity, ppv, npv, predicted_positive_rate).
        # Skipping discrimination/calibration metrics avoids refitting a
        # statsmodels Logit model on every resample.
        return fm.threshold_metrics(y_true[idx], y_pred[idx])

    # point estimate on the full (non-resampled) data
    m_a0, m_b0 = _metric(idx_a), _metric(idx_b)
    if prevalence_adjust:
        prev0 = target_prevalence
        if prev0 is None:
            prev0 = (m_a0["n_pos"] + m_b0["n_pos"]) / (m_a0["n"] + m_b0["n"])
        adj_a0 = ppv_npv_at_prevalence(m_a0["sensitivity"], m_a0["specificity"], prev0)
        adj_b0 = ppv_npv_at_prevalence(m_b0["sensitivity"], m_b0["specificity"], prev0)
        point = adj_a0[metric_key] - adj_b0[metric_key]
    else:
        point = m_a0[metric_key] - m_b0[metric_key]

    boot_gaps = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(idx_a, size=len(idx_a), replace=True)
        sb = rng.choice(idx_b, size=len(idx_b), replace=True)
        m_a, m_b = _metric(sa), _metric(sb)

        if prevalence_adjust:
            prev = target_prevalence
            if prev is None:
                prev = (m_a["n_pos"] + m_b["n_pos"]) / (m_a["n"] + m_b["n"])
            adj_a = ppv_npv_at_prevalence(m_a["sensitivity"], m_a["specificity"], prev)
            adj_b = ppv_npv_at_prevalence(m_b["sensitivity"], m_b["specificity"], prev)
            boot_gaps[i] = adj_a[metric_key] - adj_b[metric_key]
        else:
            boot_gaps[i] = m_a[metric_key] - m_b[metric_key]

    if n_boot:
        alpha = (1 - ci) / 2
        ci_low = np.nanquantile(boot_gaps, alpha)
        ci_high = np.nanquantile(boot_gaps, 1 - alpha)
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {"point": point, "ci_low": ci_low, "ci_high": ci_high, "boot": boot_gaps}


# ---------------------------------------------------------------------------
# 5. Bootstrap confidence intervals for discrimination (AUROC)
# ---------------------------------------------------------------------------
#
# AUROC is a rank statistic over a whole sample rather than a count-based
# metric, so it needs its own resampling routine: the count-based helpers above
# deliberately compute only `threshold_metrics` to avoid refitting anything per
# replicate.
#
# Conventions match the gap bootstraps above exactly -- percentile interval,
# `np.nanquantile` at alpha/2 and 1 - alpha/2, `np.random.default_rng(seed)`,
# and the same replicate count -- so all uncertainty in the study is reported
# on one convention.
#
# Resampling unit:
#   * cluster_ids is None -> observations are resampled (CDC, BRFSS, NHANES,
#     CCHS: one row per respondent).
#   * cluster_ids given    -> clusters (patients) are resampled and every
#     selected encounter of a sampled patient is retained (Diabetes-130).
#
# A replicate whose resample contains a single outcome class yields an
# undefined AUROC; it is recorded as NaN and excluded by `nanquantile`, and the
# count is returned so that the discarded fraction is visible.

BOOTSTRAP_CI_METHOD = "percentile"
AUROC_UNIT_OBSERVATION = "observation"
AUROC_UNIT_CLUSTER = "patient_cluster"


def _auroc_or_nan(y_true, y_prob, idx):
    labels = y_true[idx]
    if labels.min() == labels.max():
        return np.nan
    return float(roc_auc_score(labels, y_prob[idx]))


def _cluster_index_pool(positions, cluster_ids):
    """Map each cluster present in `positions` to its member positions."""
    by_cluster = {}
    for position in positions:
        by_cluster.setdefault(cluster_ids[position], []).append(position)
    keys = np.array(list(by_cluster), dtype=object)
    return keys, {key: np.asarray(rows) for key, rows in by_cluster.items()}


def _resample_positions(rng, positions, keys, pool):
    if keys is None:
        return rng.choice(positions, size=len(positions), replace=True)
    sampled = rng.choice(len(keys), size=len(keys), replace=True)
    return np.concatenate([pool[keys[j]] for j in sampled])


def bootstrap_auroc_ci(y_true, y_prob, mask=None, cluster_ids=None,
                       n_boot=1000, seed=42, ci=0.95):
    """Percentile bootstrap CI for AUROC on one sample.

    `mask` restricts the computation to a subgroup; omit it for the whole
    held-out partition.

    Observation-level only. Passing `cluster_ids` raises: clustered
    uncertainty goes through uncertainty.bootstrap_comparison so that every
    statistic for a comparison shares one joint patient resampling.
    """
    if cluster_ids is not None:
        _refuse_clustered("bootstrap_auroc_ci(cluster_ids=...)")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    positions = (
        np.arange(len(y_true)) if mask is None else np.where(np.asarray(mask))[0]
    )

    if cluster_ids is None:
        keys = pool = None
        unit = AUROC_UNIT_OBSERVATION
        n_units = int(len(positions))
    else:
        cluster_ids = np.asarray(cluster_ids)
        keys, pool = _cluster_index_pool(positions, cluster_ids)
        unit = AUROC_UNIT_CLUSTER
        n_units = int(len(keys))

    point = _auroc_or_nan(y_true, y_prob, positions)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot)
    for i in range(n_boot):
        replicates[i] = _auroc_or_nan(
            y_true, y_prob, _resample_positions(rng, positions, keys, pool)
        )

    if n_boot:
        alpha = (1 - ci) / 2
        ci_low = float(np.nanquantile(replicates, alpha))
        ci_high = float(np.nanquantile(replicates, 1 - alpha))
    else:
        ci_low = ci_high = np.nan

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": float(ci),
        "method": BOOTSTRAP_CI_METHOD,
        "n_boot": int(n_boot),
        "resampling_unit": unit,
        "n_observations": int(len(positions)),
        "n_units": n_units,
        "n_undefined_replicates": int(np.sum(np.isnan(replicates))),
        "boot": replicates,
    }


def bootstrap_auroc_gap_ci(y_true, y_prob, mask_a, mask_b, cluster_ids=None,
                           n_boot=1000, seed=42, ci=0.95):
    """Percentile bootstrap CI for the AUROC gap (group A - group B).

    Uses the same subgroup-separate convention as :func:`bootstrap_gap_ci`:
    each subgroup is resampled to its own original size, the metric is
    recomputed within each, and the gap is the difference. That convention is
    appropriate only where rows are independent.

    Observation-level only. Passing `cluster_ids` raises.
    """
    if cluster_ids is not None:
        _refuse_clustered("bootstrap_auroc_gap_ci(cluster_ids=...)")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    idx_a = np.where(np.asarray(mask_a))[0]
    idx_b = np.where(np.asarray(mask_b))[0]

    if cluster_ids is None:
        keys_a = pool_a = keys_b = pool_b = None
        unit = AUROC_UNIT_OBSERVATION
    else:
        cluster_ids = np.asarray(cluster_ids)
        keys_a, pool_a = _cluster_index_pool(idx_a, cluster_ids)
        keys_b, pool_b = _cluster_index_pool(idx_b, cluster_ids)
        unit = AUROC_UNIT_CLUSTER

    point = _auroc_or_nan(y_true, y_prob, idx_a) - _auroc_or_nan(
        y_true, y_prob, idx_b
    )

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot)
    for i in range(n_boot):
        ia = _resample_positions(rng, idx_a, keys_a, pool_a)
        ib = _resample_positions(rng, idx_b, keys_b, pool_b)
        replicates[i] = _auroc_or_nan(y_true, y_prob, ia) - _auroc_or_nan(
            y_true, y_prob, ib
        )

    if n_boot:
        alpha = (1 - ci) / 2
        ci_low = float(np.nanquantile(replicates, alpha))
        ci_high = float(np.nanquantile(replicates, 1 - alpha))
    else:
        ci_low = ci_high = np.nan

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": float(ci),
        "method": BOOTSTRAP_CI_METHOD,
        "n_boot": int(n_boot),
        "resampling_unit": unit,
        "n_undefined_replicates": int(np.sum(np.isnan(replicates))),
        "boot": replicates,
    }
