"""Load a frozen model artifact and score observations with it.

This is the worked example for TRIPOD+AI item 22. It demonstrates that the
serialized artifacts are sufficient to generate predictions for new
observations, and it *verifies* that by recomputing predictions and comparing
them against the frozen ones, rather than by reading a status field.

Two modes:

``verify``
    For every artifact: load it, rebuild the held-out predictor rows from the
    source dataset, score them through the loaded pipeline, and compare the
    newly computed probabilities against the persisted ones. Any deviation
    beyond the tolerance is a failure. Verification needs the source datasets,
    which are not redistributed; without them the script reports that it could
    not verify rather than claiming success.

``score``
    Score a CSV of new observations with one artifact.

Usage:
    python scripts/score_with_frozen_model.py verify --frozen-dir <dir>
    python scripts/score_with_frozen_model.py score --model <artifact.joblib> \
        --input new_rows.csv --output scored.csv
    python scripts/score_with_frozen_model.py inputs --model <artifact.joblib>

**Only raw predictor columns are required to score.** The outcome, the
demographic subgroup label, and any source identifier are not inputs to the
model and are not required; `inputs` prints the exact list. The artifact is a
Python pickle, so load it only from a source you trust: unpickling executes
code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Recomputed probabilities must match the persisted ones to this tolerance.
MATCH_TOLERANCE = 1e-12

# Which loader rebuilds each comparison's source frame. The demographic banding
# the age and race runners apply is a row FILTER, not a predictor, so scoring
# needs only the loader plus the held-out row identifiers.
SLUG_LOADER = {
    "sex_cdc": "load_cdc_diabetes",
    "sex_diabetes130": "load_diabetes130",
    "sex_brfss": "load_brfss",
    "sex_nhanes": "load_nhanes",
    "sex_cchs": "load_cchs",
    "age_cdc_diabetes_brfss_2015": "load_cdc_diabetes",
    "age_diabetes_130_readmission": "load_diabetes130",
    "age_brfss_2022_heart_disease": "load_brfss",
    "age_nhanes_2017_18_diabetes": "load_nhanes",
    "age_cchs_2019_20_diabetes": "load_cchs",
    "race_diabetes_130_readmission": "load_diabetes130",
    "race_brfss_2022_heart_disease": "load_brfss",
    "race_nhanes_2017_18_diabetes": "load_nhanes",
    "race_brfss_2022_heart_disease_hispanic_vs_white": "load_brfss",
    "race_brfss_2022_heart_disease_multiracial_vs_white": "load_brfss",
}


def load_artifact(path):
    """Load one frozen model artifact and check it carries what scoring needs."""
    import joblib

    payload = joblib.load(path)
    required = (
        "preprocessor", "estimator", "calibrated_model", "feature_order",
        "required_predictor_columns", "operating_threshold", "selected_family",
        "sklearn_version",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"{Path(path).name}: artifact is missing {missing}")
    return payload


def required_predictor_columns(payload):
    """The raw columns a caller must supply to score. Nothing else is needed."""
    return list(payload["required_predictor_columns"])


def score_frame(payload, frame):
    """Predicted probability and thresholded prediction for raw predictor rows.

    The calibrated model is a pipeline whose first step is the training-fitted
    preprocessing, so it consumes RAW predictor columns. Callers must not
    pre-transform, and must not be asked for the outcome or the subgroup label.
    """
    needed = required_predictor_columns(payload)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"input is missing {len(missing)} required predictor column(s): "
            f"{missing[:10]}"
        )
    probability = payload["calibrated_model"].predict_proba(frame[needed])[:, 1]
    threshold = float(payload["operating_threshold"])
    return pd.DataFrame({
        "prob_calibrated": probability,
        "y_pred_at_threshold": (probability >= threshold).astype(int),
        "operating_threshold": threshold,
        "selected_family": payload["selected_family"],
    })


def _rebuild_held_out_predictors(slug, row_ids):
    """Reconstruct the held-out predictor rows for one comparison.

    Deterministic: it loads the dataset and selects the recorded held-out row
    identifiers. No model is fitted and no split is recomputed.
    """
    from src import datasets

    loader_name = SLUG_LOADER.get(slug)
    if loader_name is None:
        raise RuntimeError(f"{slug}: no loader mapping")
    frame = getattr(datasets, loader_name)()["df"]
    missing = [r for r in row_ids if r not in frame.index]
    if missing:
        raise RuntimeError(
            f"{slug}: {len(missing)} held-out row id(s) absent from the "
            "reloaded dataset"
        )
    return frame.loc[row_ids]


def verify(frozen_dir):
    """Recompute every artifact's predictions and compare with the frozen ones."""
    frozen_dir = Path(frozen_dir)
    artifacts = sorted(frozen_dir.glob("*_model.joblib"))
    if not artifacts:
        print(f"no model artifacts under {frozen_dir}")
        return 1

    failures = unverifiable = 0
    for artifact in artifacts:
        slug = artifact.name[: -len("_model.joblib")]
        predictions_path = frozen_dir / f"{slug}_predictions.csv"
        if not predictions_path.exists():
            print(f"  {slug:52s} FAIL  no frozen predictions to compare against")
            failures += 1
            continue

        saved = pd.read_csv(predictions_path)
        if "row_id" not in saved.columns:
            print(f"  {slug:52s} SKIP  public export carries no source row ids; "
                  "verification needs the internal run outputs")
            unverifiable += 1
            continue

        payload = load_artifact(artifact)
        try:
            rows = _rebuild_held_out_predictors(slug, saved["row_id"].tolist())
        except FileNotFoundError:
            print(f"  {slug:52s} SKIP  source dataset unavailable; not verified")
            unverifiable += 1
            continue

        recomputed = score_frame(payload, rows)["prob_calibrated"].to_numpy()
        expected = saved["prob_calibrated"].to_numpy()
        deviation = float(np.max(np.abs(recomputed - expected)))
        agree = deviation <= MATCH_TOLERANCE

        pred_expected = saved["y_pred_at_threshold"].to_numpy()
        pred_recomputed = (
            recomputed >= float(payload["operating_threshold"])
        ).astype(int)
        preds_agree = bool(np.array_equal(pred_expected, pred_recomputed))

        ok = agree and preds_agree
        failures += not ok
        print(f"  {slug:52s} {'OK  ' if ok else 'FAIL'}  "
              f"{payload['selected_family']:22s} rows={len(saved):6d} "
              f"inputs={len(required_predictor_columns(payload)):2d} "
              f"max|dev|={deviation:.2e}")

    checked = len(artifacts) - unverifiable
    print(f"\n{checked - failures}/{checked} artifacts recomputed and matched"
          + (f"; {unverifiable} not verifiable here" if unverifiable else ""))
    return 1 if failures else 0


def score(model_path, input_path, output_path):
    payload = load_artifact(Path(model_path))
    frame = pd.read_csv(input_path)
    scored = score_frame(payload, frame)
    scored.to_csv(output_path, index=False)
    print(f"scored {len(scored)} row(s) -> {output_path}")
    print(f"  family    : {payload['selected_family']}")
    print(f"  threshold : {payload['operating_threshold']}")
    print(f"  sklearn   : {payload['sklearn_version']}")
    return 0


def inputs(model_path):
    payload = load_artifact(Path(model_path))
    needed = required_predictor_columns(payload)
    print(f"{payload['slug']}: {len(needed)} required predictor column(s)")
    for name in needed:
        print(f"  {name}")
    print("\nNot required: the outcome, the demographic subgroup label, and any "
          "source identifier. They are not model inputs.")
    schema = payload.get("preprocessing_schema", {})
    if schema.get("categorical_sources"):
        print("\nDeterministic recoding applied inside the pipeline:")
        for spec in schema["categorical_sources"]:
            print(f"  {spec['source']} -> one-hot, reference level "
                  f"{spec.get('reference_level')!r}")
    if schema.get("training_medians"):
        print(f"\n{len(schema['training_medians'])} numeric column(s) are "
              "median-imputed using training-derived values carried in the "
              "artifact.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    v = sub.add_parser("verify", help="recompute and compare every artifact")
    v.add_argument("--frozen-dir", required=True)

    s = sub.add_parser("score", help="score new observations")
    s.add_argument("--model", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)

    i = sub.add_parser("inputs", help="print the required predictor columns")
    i.add_argument("--model", required=True)

    args = parser.parse_args()
    if args.mode == "verify":
        return verify(args.frozen_dir)
    if args.mode == "inputs":
        return inputs(args.model)
    return score(args.model, args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
