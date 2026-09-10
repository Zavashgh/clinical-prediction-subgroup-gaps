"""
uncertainty.py
==============
Unified resampling for one comparison's held-out evaluation.

Why this module exists
----------------------
Uncertainty for a comparison was previously computed by several independent
routines: Wilson intervals for subgroup threshold metrics, a subgroup-separate
bootstrap for gaps, and a separate bootstrap for AUROC. For a dataset with
repeated encounters per patient that combination is not coherent:

* Wilson intervals treat encounters as independent Bernoulli trials, which they
  are not when one patient contributes several;
* resampling group A and group B *separately* draws an independent multiplicity
  for a patient who appears under more than one demographic label, breaking the
  dependence between the two subgroup estimates that the gap is computed from.

This module replaces all of that with a single pass over replicates. For a
clustered dataset the resampling unit is the PATIENT and each patient is drawn
**once per replicate**, with the resulting multiplicity applied to every one of
that patient's encounters regardless of which subgroup each encounter falls in.
Subgroup metrics, gaps, discrimination, and calibration are then all computed
from the same resampled encounter set, so they share one coherent resampling
distribution.

Unclustered datasets keep the established subgroup-separate row-level
convention, which preserves each subgroup's sample size in every replicate.

Everything else matches the study's existing convention: percentile limits at
alpha/2 and 1 - alpha/2, ``numpy.random.default_rng(seed)``, and the same
replicate count.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from . import metrics as fm
from .adjustments import ppv_npv_at_prevalence

CI_METHOD = "percentile"
UNIT_OBSERVATION = "observation"
UNIT_CLUSTER = "patient_cluster"

RESAMPLING_JOINT_CLUSTER = "joint_patient_cluster_whole_test_set"
RESAMPLING_SEPARATE_ROWS = "row_level_within_subgroup_resampling"

# Threshold metrics reported per subgroup and as gaps.
THRESHOLD_METRICS = (
    "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv",
    "predicted_positive_rate", "prevalence",
)
PREVALENCE_ADJUSTED_METRICS = ("ppv", "npv", "predicted_positive_rate")


# The calibration solver lives in metrics.py so that the point estimates and the
# bootstrap replicates are produced by exactly the same code path. Two
# implementations of one statistic is how they drift apart.
calibration_logit = fm.calibration_logit
fit_calibration_intercept_slope = fm.fit_calibration_intercept_slope
CALIBRATION_LOGIT_EPS = fm.CALIBRATION_LOGIT_EPS


def calibration_statistics(y_true, y_prob):
    """Brier score plus calibration intercept and slope with an explicit status."""
    return fm.calibration_metrics(y_true, y_prob)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _cluster_pool(positions, cluster_ids):
    by_cluster = {}
    for position in positions:
        by_cluster.setdefault(cluster_ids[position], []).append(position)
    keys = np.array(list(by_cluster), dtype=object)
    return keys, {k: np.asarray(v) for k, v in by_cluster.items()}


def joint_cluster_positions(rng, keys, pool):
    """Draw clusters once and keep every encounter of each drawn cluster.

    A patient sampled twice contributes all of their encounters twice, in every
    subgroup those encounters belong to. This is what preserves dependence for a
    patient recorded under more than one demographic label.
    """
    drawn = rng.choice(len(keys), size=len(keys), replace=True)
    return np.concatenate([pool[keys[j]] for j in drawn])


def _auroc(y_true, y_prob, idx):
    if len(idx) == 0:
        return np.nan
    labels = y_true[idx]
    if labels.min() == labels.max():
        return np.nan
    return float(roc_auc_score(labels, y_prob[idx]))


def _percentile_interval(values, ci):
    values = np.asarray(values, dtype=float)
    valid = np.count_nonzero(np.isfinite(values))
    if valid == 0:
        return np.nan, np.nan, 0
    alpha = (1.0 - ci) / 2.0
    return (float(np.nanquantile(values, alpha)),
            float(np.nanquantile(values, 1.0 - alpha)),
            int(valid))


def bootstrap_comparison(
    y_true, y_prob, y_pred, mask_a, mask_b, *,
    cluster_ids=None, n_boot=1000, seed=42, ci=0.95,
    target_prevalence=None,
):
    """All uncertainty for one comparison, from a single set of replicates.

    With ``cluster_ids`` the replicate is a joint draw of patients over the
    whole held-out set; without it, group A and group B are resampled
    separately at the row level, preserving each subgroup's size.

    Returns a dict with ``subgroup_a``, ``subgroup_b``, ``gaps``,
    ``prevalence_adjusted_gaps``, ``discrimination``, ``calibration`` and a
    ``design`` block recording the resampling unit, replicate count, and how
    many replicates yielded a usable value for each statistic.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred)
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]
    idx_all = np.where(mask_a | mask_b)[0]

    clustered = cluster_ids is not None
    if clustered:
        cluster_ids = np.asarray(cluster_ids)
        keys_all, pool_all = _cluster_pool(idx_all, cluster_ids)
        unit, design = UNIT_CLUSTER, RESAMPLING_JOINT_CLUSTER
        n_units = int(len(keys_all))
        spanning = int(sum(
            1 for k in keys_all
            if len({("A" if mask_a[p] else "B") for p in pool_all[k]}) > 1
        ))
    else:
        unit, design = UNIT_OBSERVATION, RESAMPLING_SEPARATE_ROWS
        n_units = int(len(idx_all))
        spanning = 0

    logit_all = calibration_logit(y_prob)

    def point_block(ia, ib):
        """Every reported statistic for one (possibly resampled) index pair."""
        ma = fm.threshold_metrics(y_true[ia], y_pred[ia])
        mb = fm.threshold_metrics(y_true[ib], y_pred[ib])
        block = {"a": {}, "b": {}, "gap": {}, "adj": {}}
        for key in THRESHOLD_METRICS:
            block["a"][key] = ma[key]
            block["b"][key] = mb[key]
            block["gap"][key] = ma[key] - mb[key]
        prevalence = target_prevalence
        if prevalence is None:
            total = ma["n"] + mb["n"]
            prevalence = (ma["n_pos"] + mb["n_pos"]) / total if total else np.nan
        adj_a = ppv_npv_at_prevalence(ma["sensitivity"], ma["specificity"], prevalence)
        adj_b = ppv_npv_at_prevalence(mb["sensitivity"], mb["specificity"], prevalence)
        for key in PREVALENCE_ADJUSTED_METRICS:
            block["adj"][key] = adj_a[key] - adj_b[key]

        iall = np.concatenate([ia, ib])
        block["auroc"] = {
            "overall": _auroc(y_true, y_prob, iall),
            "a": _auroc(y_true, y_prob, ia),
            "b": _auroc(y_true, y_prob, ib),
        }
        block["auroc"]["gap"] = block["auroc"]["a"] - block["auroc"]["b"]

        block["calib"] = {}
        for name, idx in (("overall", iall), ("a", ia), ("b", ib)):
            y_s, p_s = y_true[idx], y_prob[idx]
            intercept, slope, status = fit_calibration_intercept_slope(
                y_s, logit_all[idx]
            )
            block["calib"][name] = {
                "brier": float(np.mean((p_s - y_s) ** 2)) if len(idx) else np.nan,
                "intercept": intercept,
                "slope": slope,
                "status": status,
            }
        return block

    point = point_block(idx_a, idx_b)

    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot):
        if clustered:
            drawn = joint_cluster_positions(rng, keys_all, pool_all)
            ia = drawn[mask_a[drawn]]
            ib = drawn[mask_b[drawn]]
        else:
            ia = rng.choice(idx_a, size=len(idx_a), replace=True)
            ib = rng.choice(idx_b, size=len(idx_b), replace=True)
        replicates.append(point_block(ia, ib))

    def summarize(extract):
        values = [extract(r) for r in replicates]
        low, high, valid = _percentile_interval(values, ci)
        return {
            "ci_low": low, "ci_high": high,
            "n_valid_replicates": valid,
            "n_failed_replicates": int(n_boot - valid),
        }

    out = {
        "design": {
            "resampling_unit": unit,
            "resampling_design": design,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "ci_level": float(ci),
            "ci_method": CI_METHOD,
            "n_resampling_units": n_units,
            "n_observations": int(len(idx_all)),
            "n_units_spanning_subgroups": spanning,
        },
        "subgroup_a": {}, "subgroup_b": {}, "gaps": {},
        "prevalence_adjusted_gaps": {}, "discrimination": {}, "calibration": {},
    }
    for key in THRESHOLD_METRICS:
        out["subgroup_a"][key] = {"point": point["a"][key],
                                  **summarize(lambda r, k=key: r["a"][k])}
        out["subgroup_b"][key] = {"point": point["b"][key],
                                  **summarize(lambda r, k=key: r["b"][k])}
        out["gaps"][key] = {"point": point["gap"][key],
                            **summarize(lambda r, k=key: r["gap"][k])}
    for key in PREVALENCE_ADJUSTED_METRICS:
        out["prevalence_adjusted_gaps"][key] = {
            "point": point["adj"][key],
            **summarize(lambda r, k=key: r["adj"][k]),
        }
    for name in ("overall", "a", "b", "gap"):
        out["discrimination"][name] = {
            "point": point["auroc"][name],
            **summarize(lambda r, n=name: r["auroc"][n]),
        }
    for name in ("overall", "a", "b"):
        block = {}
        for stat in ("brier", "intercept", "slope"):
            block[stat] = {
                "point": point["calib"][name][stat],
                **summarize(lambda r, n=name, s=stat: r["calib"][n][s]),
            }
        block["status"] = point["calib"][name]["status"]
        block["replicate_status_counts"] = _status_counts(replicates, name)
        out["calibration"][name] = block
    return out


def _status_counts(replicates, name):
    counts = {}
    for r in replicates:
        status = r["calib"][name]["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
