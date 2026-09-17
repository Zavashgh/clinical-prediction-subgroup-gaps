# Prevalence, Thresholds, and Subgroup Performance Gaps in Clinical Prediction Models: A Multi-Dataset Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22694874.svg)](https://doi.org/10.5281/zenodo.22694874)

Reproducibility package for the JAMIA Open manuscript of the same title:
analysis code, frozen held-out predictions, fitted model artifacts, and
reference outputs for the final corrected analysis.

| | |
|---|---|
| Release | [v1.1.0](https://github.com/Zavashgh/clinical-prediction-subgroup-gaps/releases/tag/v1.1.0) |
| Version DOI | [10.5281/zenodo.22694874](https://doi.org/10.5281/zenodo.22694874) |
| Concept DOI (all versions) | [10.5281/zenodo.22679528](https://doi.org/10.5281/zenodo.22679528) |
| Archived record | https://zenodo.org/records/22694874 |
| Production run | `20260909T234626Z_5b0145ea3bb0_e89e8377` |
| Licence | MIT (code) |

Cite the **version DOI** when referring to the analysis reported in the
manuscript.

## What is here

- `src/` — the analysis library
- `notebooks/`, `run_age_decomposition.py`, `run_race_decomposition.py` — the
  analyses
- `scripts/` — production orchestration, reporting sources, run comparison,
  `score_with_frozen_model.py` for scoring new observations, and
  `verify_run_integrity.py` for checking the run's attestations
- `tests/`, `validation_reference/` — test suites
- `reference_outputs/` — the run's result tables and figures, run manifest,
  provenance, and post-run artifact amendment
- `frozen_predictions/` — row-level held-out predictions for all 15 fitted
  comparisons, with per-comparison discrimination and calibration tables
- `model_bundle_manifest.csv` — checksum, size, required predictor columns, and
  software versions for every fitted model artifact
- `RELEASE_INVENTORY.csv` — where every output of the production run went

## Fitted model artifacts

The fitted preprocessing, estimator, and calibration objects for all 15
comparisons are in `fitted_models_v1_1_0.tar.gz` (158 MB compressed, 673 MB
extracted), distributed with the [Zenodo record](https://zenodo.org/records/22694874) and the
[GitHub release](https://github.com/Zavashgh/clinical-prediction-subgroup-gaps/releases/tag/v1.1.0) rather than committed here, because two artifacts
exceed the 100 MB per-file limit of Git hosting. `SHA256SUMS.txt` accompanies
it.

```
tar -xzf fitted_models_v1_1_0.tar.gz
python scripts/score_with_frozen_model.py inputs --model fitted_models_v1_1_0/sex_cdc_model.joblib
python scripts/score_with_frozen_model.py score  --model fitted_models_v1_1_0/sex_cdc_model.joblib \
    --input new_rows.csv --output scored.csv
```

Only the **raw predictor columns** listed by `inputs` are required. The
outcome, the demographic subgroup label, and identifiers are not model inputs.

**Requirements:** Python 3.11.9 with the package versions in `requirements.txt`
and `reference_outputs/run_manifest.json`. Scoring runs on a CPU and needs no
GPU or other specialised hardware.

**Conditions of use:** the artifacts are Python pickles. Load them only with the
recorded package versions, and only from a trusted source, since unpickling
executes code. The frozen prediction files are the portable record of the
reported evaluation.

## Reproducing the analysis

```
pip install -r requirements.txt
python scripts/run_production.py run-jama --execute-production
```

Source datasets are **not redistributed**. Obtain them from the providers listed
in `data/README.md` and place them in `data/` before running.

To validate the package without the source data:

```
python tests/release_suite.py
```

## Verifying the reference outputs

The run carries two attestations. `reference_outputs/run_manifest.json` attests
to the run at completion. `reference_outputs/POST_RUN_ARTIFACT_AMENDMENT.json`
attests to the 15 fitted artifacts and 15 companion manifests whose
scoring-interface metadata was corrected afterwards; no model was refitted and
no probability changed.

```
python scripts/verify_run_integrity.py --run-dir <run>
```

expects 85 outputs matching the run manifest, 30 matching the amendment, and 0
unexplained mismatches.

## Privacy of the released predictions

Row identifiers in `frozen_predictions/` are opaque release-local tokens
(`row_000001`, ...); source row indices are not published. For Diabetes-130 the
cluster identifier is an anonymous token (`P000001`, ...) that preserves patient
grouping exactly, which is all the joint patient bootstrap needs; the source
patient identifier is not published.

## Method summary

- Six model families; selection by five-fold cross-validated AUROC on training
  data only.
- Data-derived preprocessing is fitted inside each cross-validation and
  calibration fold; each calibrated fold model carries its own preprocessor.
- Five-fold isotonic fold-ensemble calibration; operating threshold at the
  training-set outcome prevalence.
- Diabetes-130 is patient-level throughout, with uncertainty from a joint
  patient-cluster bootstrap. The four one-row-per-respondent datasets use
  observation-level resampling, with Wilson intervals for subgroup proportions.
- AUROC and calibration (Brier score, intercept, slope) are reported with 95%
  bootstrap intervals.

## Version history

Release v1.0.0 ([10.5281/zenodo.22679529](https://doi.org/10.5281/zenodo.22679529))
is an earlier analysis retained as a historical record. It does not archive the
analysis reported in the manuscript; cite v1.1.0.

## Citation

See `CITATION.cff`. The manuscript is not yet published or accepted, and this
repository makes no such claim.
