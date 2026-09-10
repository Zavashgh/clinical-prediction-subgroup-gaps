"""
frozen_outputs.py
=================
Persist the held-out evaluation of a finished comparison so that later
reporting never needs to refit a model.

Two artifacts are written per comparison, both under ``results/frozen/``:

``<slug>_predictions.csv``
    One row per held-out observation, carrying the true outcome, the
    calibrated and uncalibrated probabilities, the binary prediction at the
    manuscript operating threshold, the subgroup label, and -- where the
    dataset has repeated encounters -- the grouping identifier needed for
    clustered inference. Everything required to recompute any threshold,
    discrimination, or calibration statistic is in this file.

``<slug>_model.joblib``
    The fitted training-only preprocessor, the fitted selected estimator, and
    the fitted calibrated fold-ensemble, plus the feature ordering and
    threshold metadata needed to score a new observation. Serialization is
    verified by reloading the file and reproducing the exported calibrated
    probabilities exactly; the verification outcome is recorded rather than
    assumed, and a family that cannot be serialized portably is reported
    instead of being forced.

No source predictor values are exported. The only identifier written is the
grouping key already required for clustered resampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "dataset",
    "comparison_type",
    "comparison",
    "row_id",
    "cluster_id",
    "subgroup",
    "y_true",
    "prob_calibrated",
    "prob_uncalibrated",
    "y_pred_at_threshold",
    "selected_family",
    "operating_threshold",
)

FROZEN_DIRNAME = "frozen"

# Cluster labels are released as arbitrary deterministic tokens rather than the
# source patient identifier. Grouping is preserved exactly -- two encounters
# share a label if and only if they share a patient -- which is all that
# clustered inference needs. The source identifier is not required for any
# reported analysis, so it is not exported.
CLUSTER_LABEL_PREFIX = "P"


def anonymize_clusters(values):
    """Map cluster values to deterministic ``P000001``-style labels.

    Labels are assigned in order of the sorted distinct values, so the mapping
    is reproducible from the same held-out set and carries no information about
    the source identifier beyond the grouping itself.
    """
    import pandas as _pd

    series = _pd.Series(values)
    distinct = sorted(series.dropna().unique().tolist(), key=lambda v: str(v))
    lookup = {
        value: f"{CLUSTER_LABEL_PREFIX}{i + 1:06d}"
        for i, value in enumerate(distinct)
    }
    return series.map(lookup).to_numpy(), lookup

# Reloading must reproduce the exported probabilities to this tolerance. It is
# exact equality in practice; the tolerance only absorbs float round-tripping.
SERIALIZATION_TOLERANCE = 1e-12


def _frozen_dir(results_dir) -> Path:
    path = Path(results_dir) / FROZEN_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_frozen_predictions(
    result,
    *,
    dataset,
    comparison_type,
    comparison,
    cluster_ids=None,
) -> pd.DataFrame:
    """Assemble the row-level held-out table for one finished comparison.

    `cluster_ids` is any object indexable by the original dataframe index; the
    held-out rows are looked up by ``result["test_index"]`` so the exported
    grouping key lines up with the exported predictions by construction.
    """
    test_index = np.asarray(result["test_index"])
    n = len(test_index)

    if cluster_ids is None:
        clusters = np.full(n, "", dtype=object)
    else:
        if isinstance(cluster_ids, pd.Series):
            clusters = cluster_ids.reindex(pd.Index(test_index)).to_numpy()
        else:
            clusters = np.asarray(cluster_ids)[: len(test_index)]
        if pd.isnull(pd.Series(clusters)).any():
            raise ValueError(
                f"{dataset}/{comparison}: cluster identifiers are missing for "
                "some held-out rows; clustered inference would be unreliable"
            )
        clusters, _ = anonymize_clusters(clusters)

    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "comparison_type": comparison_type,
            "comparison": comparison,
            "row_id": test_index,
            "cluster_id": clusters,
            "subgroup": np.asarray(result["g_test"]),
            "y_true": np.asarray(result["y_test"]).astype(int),
            "prob_calibrated": np.asarray(result["prob"], dtype=float),
            "prob_uncalibrated": np.asarray(
                result["prob_uncalibrated"], dtype=float
            ),
            "y_pred_at_threshold": np.asarray(result["pred"]).astype(int),
            "selected_family": result["best_model_name"],
            "operating_threshold": float(result["decision_threshold"]),
        },
        columns=list(PREDICTION_COLUMNS),
    )

    _assert_export_matches_result(frame, result, f"{dataset}/{comparison}")
    return frame


def _assert_export_matches_result(frame, result, context):
    """The export must be the evaluated rows themselves, not a rebuild of them."""
    if len(frame) != int(result["n_test"]):
        raise RuntimeError(
            f"{context}: exported {len(frame)} rows but the model was evaluated "
            f"on {result['n_test']}"
        )
    if not np.array_equal(
        frame["y_true"].to_numpy(), np.asarray(result["y_test"]).astype(int)
    ):
        raise RuntimeError(f"{context}: exported outcomes differ from the evaluated ones")
    if not np.array_equal(
        frame["prob_calibrated"].to_numpy(), np.asarray(result["prob"], dtype=float)
    ):
        raise RuntimeError(
            f"{context}: exported probabilities differ from the evaluated ones"
        )
    threshold = float(result["decision_threshold"])
    expected = (frame["prob_calibrated"].to_numpy() >= threshold).astype(int)
    if not np.array_equal(frame["y_pred_at_threshold"].to_numpy(), expected):
        raise RuntimeError(
            f"{context}: exported binary predictions are not the calibrated "
            "probabilities thresholded at the operating threshold"
        )


def write_frozen_predictions(
    result,
    *,
    slug,
    dataset,
    comparison_type,
    comparison,
    results_dir,
    cluster_ids=None,
) -> Path:
    """Write ``<slug>_predictions.csv`` and return its path."""
    frame = build_frozen_predictions(
        result,
        dataset=dataset,
        comparison_type=comparison_type,
        comparison=comparison,
        cluster_ids=cluster_ids,
    )
    path = _frozen_dir(results_dir) / f"{slug}_predictions.csv"
    frame.to_csv(path, index=False)
    return path


def write_fitted_objects(result, *, slug, results_dir) -> dict:
    """Serialize the fitted objects and verify they reproduce the predictions.

    Returns a metadata dict recording whether serialization succeeded and
    whether the reloaded objects reproduced the exported probabilities. A
    failure is reported, never silently swallowed and never left unverified.
    """
    path = _frozen_dir(results_dir) / f"{slug}_model.joblib"
    metadata = {
        "slug": slug,
        "artifact": f"{FROZEN_DIRNAME}/{path.name}",
        "selected_family": result["best_model_name"],
        "operating_threshold": float(result["decision_threshold"]),
        "calibration_protocol": result["calibration_protocol"],
        "feature_order": list(result["feature_names"]),
        "serialized": False,
        "verified": False,
        "status": "not_attempted",
        "sha256": None,
    }

    try:
        import joblib
        import sklearn
    except Exception as error:  # pragma: no cover - joblib ships with sklearn
        metadata["status"] = f"unavailable: {type(error).__name__}: {error}"
        return metadata

    payload = {
        "schema_version": "1.0.0",
        "slug": slug,
        "preprocessor": result["preprocessor"],
        # Only the RAW PREDICTOR columns are required to score. The outcome,
        # the demographic subgroup label, and any source identifier are not
        # model inputs and must not be advertised as required: doing so forces
        # a caller scoring genuinely new observations to supply data they do
        # not have and the model never uses.
        "required_predictor_columns": list(
            result["preprocessing_schema"].get(
                "feature_order_declared", result["feature_names"]
            )
        ),
        "metadata_schema_version": "1.1.0",
        "not_required_for_scoring": [
            "outcome / target label",
            "demographic subgroup label (unless it is itself a predictor)",
            "source record identifiers",
            "derived comparison labels",
        ],
        "preprocessing_schema": result["preprocessing_schema"],
        "estimator": result["models"][result["best_model_name"]],
        "calibrated_model": result["calibrated_model"],
        "selected_family": result["best_model_name"],
        "feature_order": list(result["feature_names"]),
        "operating_threshold": float(result["decision_threshold"]),
        "threshold_protocol": result["threshold_protocol"],
        "calibration_protocol": result["calibration_protocol"],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }

    try:
        joblib.dump(payload, path)
    except Exception as error:
        metadata["status"] = f"serialization_failed: {type(error).__name__}: {error}"
        return metadata
    metadata["serialized"] = True
    metadata["sklearn_version"] = sklearn.__version__
    metadata["joblib_version"] = joblib.__version__

    # Verify by reloading and re-scoring the SAME design matrix the run used.
    try:
        reloaded = joblib.load(path)
        replayed = reloaded["calibrated_model"].predict_proba(result["raw_test"])[:, 1]
        expected = np.asarray(result["prob"], dtype=float)
        deviation = float(np.max(np.abs(replayed - expected))) if len(expected) else 0.0
        if deviation <= SERIALIZATION_TOLERANCE:
            metadata["verified"] = True
            metadata["status"] = "verified_reproduces_frozen_predictions"
        else:
            metadata["status"] = f"verification_failed: max_abs_deviation={deviation:.3e}"
        metadata["max_abs_deviation"] = deviation
    except Exception as error:
        metadata["status"] = f"verification_error: {type(error).__name__}: {error}"

    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["size_bytes"] = path.stat().st_size
    return metadata


def performance_rows(result, *, dataset, comparison_type, comparison):
    """Overall and per-subgroup discrimination + calibration, with intervals."""
    labels = result["group_labels"]
    values = list(labels)
    calib = result["calibration_intervals"]
    rows = [
        _performance_row(
            result, dataset, comparison_type, comparison, "overall", "overall",
            result["overall_metrics"], result["auroc_ci"]["overall"],
            calib["overall"],
        )
    ]
    for side, key, value in (("a", "A", values[0]), ("b", "B", values[1])):
        rows.append(
            _performance_row(
                result, dataset, comparison_type, comparison,
                labels[value], str(value),
                result["subgroup_metrics"][key], result["auroc_ci"][key],
                calib[side],
            )
        )
    return pd.DataFrame(rows)


def _ci(block, field):
    return block[field] if block is not None else np.nan


def _performance_row(result, dataset, comparison_type, comparison,
                     group_label, group_value, metrics, auroc_ci, calib):
    design = result["resampling_design"]
    return {
        "dataset": dataset,
        "comparison_type": comparison_type,
        "comparison": comparison,
        "group": group_label,
        "group_value": group_value,
        "n": int(metrics["n"]),
        "n_events": int(metrics["n_pos"]),
        "selected_family": result["best_model_name"],
        "operating_threshold": float(result["decision_threshold"]),
        "auroc": metrics["auroc"],
        "auroc_ci_low": auroc_ci["ci_low"],
        "auroc_ci_high": auroc_ci["ci_high"],
        "auprc": metrics["auprc"],
        "brier": calib["brier"]["point"],
        "brier_ci_low": calib["brier"]["ci_low"],
        "brier_ci_high": calib["brier"]["ci_high"],
        "calib_intercept": calib["intercept"]["point"],
        "calib_intercept_ci_low": calib["intercept"]["ci_low"],
        "calib_intercept_ci_high": calib["intercept"]["ci_high"],
        "calib_slope": calib["slope"]["point"],
        "calib_slope_ci_low": calib["slope"]["ci_low"],
        "calib_slope_ci_high": calib["slope"]["ci_high"],
        "calib_status": calib["status"],
        "calib_slope_valid_replicates": calib["slope"]["n_valid_replicates"],
        "calib_slope_failed_replicates": calib["slope"]["n_failed_replicates"],
        "calib_replicate_status_counts": json.dumps(
            calib["replicate_status_counts"], sort_keys=True
        ),
        "calib_n_clipped": int(metrics["calib_n_clipped"]),
        "ece": metrics["ece"],
        "ci_method": design["ci_method"],
        "ci_level": design["ci_level"],
        "n_boot": design["n_boot"],
        "resampling_unit": design["resampling_unit"],
        "resampling_design": design["resampling_design"],
        "n_resampling_units": design["n_resampling_units"],
        "n_units_spanning_subgroups": design["n_units_spanning_subgroups"],
    }


def write_comparison_artifacts(
    result,
    *,
    slug,
    dataset,
    comparison_type,
    comparison,
    results_dir,
    cluster_ids=None,
    serialize_model=True,
) -> dict:
    """Write every frozen artifact for one comparison and return a manifest."""
    predictions_path = write_frozen_predictions(
        result,
        slug=slug,
        dataset=dataset,
        comparison_type=comparison_type,
        comparison=comparison,
        results_dir=results_dir,
        cluster_ids=cluster_ids,
    )

    performance = performance_rows(
        result, dataset=dataset, comparison_type=comparison_type,
        comparison=comparison,
    )
    performance_path = _frozen_dir(results_dir) / f"{slug}_performance.csv"
    performance.to_csv(performance_path, index=False)

    model_metadata = (
        write_fitted_objects(result, slug=slug, results_dir=results_dir)
        if serialize_model
        else {"slug": slug, "status": "not_requested",
              "serialized": False, "verified": False}
    )

    manifest = {
        "slug": slug,
        "dataset": dataset,
        "comparison_type": comparison_type,
        "comparison": comparison,
        "n_test": int(result["n_test"]),
        "selected_family": result["best_model_name"],
        "operating_threshold": float(result["decision_threshold"]),
        "split_protocol": result["split_protocol"],
        "bootstrap_protocol": result["bootstrap_protocol"],
        "calibration_protocol": result["calibration_protocol"],
        "clustered": cluster_ids is not None,
        "resampling_design": result["resampling_design"],
        "subgroup_ci_method": result["subgroup_ci_method"],
        "required_input_columns": list(result["preprocessing_schema"].get(
            "feature_order_declared", result["feature_names"])),
        "predictions": f"{FROZEN_DIRNAME}/{predictions_path.name}",
        "performance": f"{FROZEN_DIRNAME}/{performance_path.name}",
        "model": model_metadata,
    }
    manifest_path = _frozen_dir(results_dir) / f"{slug}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
