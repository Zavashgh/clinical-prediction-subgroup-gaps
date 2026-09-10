# Prevalence, Thresholds, and Subgroup Performance Gaps in Clinical Prediction Models: A Multi-Dataset Analysis

**Release candidate v1.1.0 — not yet published.**

Analysis code, frozen held-out predictions, and reference outputs for the
definitive corrected analysis.

| | |
|---|---|
| Production run | `20260909T234626Z_5b0145ea3bb0_e89e8377` |
| Analysis source commit | `5b0145ea3bb0801551f5f343c24b9366b3192a1d` |
| Candidate version | `v1.1.0` |
| DOI | **not yet minted for this version** |
| Licence | MIT (code only) |

## Relationship to the archived v1.0.0

`v1.0.0` (DOI [10.5281/zenodo.22679529](https://doi.org/10.5281/zenodo.22679529))
is a **published archival record of an earlier analysis**. It is not modified by
this candidate, and **its DOI does not identify the analysis in this
directory**. This candidate supersedes it scientifically; the two are different
analyses and should not be conflated.

The changes between them are methodological, not cosmetic:

- **Preprocessing is now fitted inside each fold.** Previously one preprocessor
  was fitted on the whole outer-training set and reused inside every
  model-selection and calibration fold, so inner validation rows contributed to
  the medians and categorical schema of the model then evaluated on them. Each
  fold now fits its own, and each fold-specific calibrated model carries it.
- **Diabetes-130 uncertainty is a joint patient-cluster bootstrap.** Subgroup
  intervals were previously encounter-level binomial intervals, and the gap
  bootstrap drew an independent multiplicity for a patient appearing under more
  than one demographic label. Patients are now drawn once per replicate across
  the whole held-out set, and that multiplicity applies to all their encounters.
- **Calibration is reported with intervals**, and an intercept or slope is
  withheld only when the fit is not numerically valid, with the reason named.
- **All 15 fitted comparisons are frozen**, including the two secondary BRFSS
  race comparisons.

Selected model families, operating thresholds, gap directions, and
interval-excludes-zero classifications are unchanged from v1.0.0.

## Contents

- `src/` — the analysis library
- `notebooks/`, `run_age_decomposition.py`, `run_race_decomposition.py` — the analyses
- `scripts/` — production orchestration, reporting sources, run comparison, and
  `score_with_frozen_model.py`, the worked scoring example. Run
  `python scripts/score_with_frozen_model.py inputs --model <artifact>` to list
  the **raw predictor columns** an artifact requires; the outcome, the subgroup
  label, and identifiers are not inputs
- `tests/`, `validation_reference/` — the test suites
- `reference_outputs/` — the run's result CSVs, manifest, and provenance
- `frozen_predictions/` — row-level held-out predictions plus per-comparison
  performance and calibration tables
- `model_bundle_manifest.csv` — checksum, size, family, and required input
  columns for all 15 fitted artifacts

## Reproducing

```
pip install -r requirements.txt
python scripts/run_production.py run-jama --execute-production
```

Source datasets are **not redistributed**; obtain them from the providers listed
in `data/README.md`.

## Validating this package

```
python tests/release_suite.py
```

## Release inventory

`RELEASE_INVENTORY.csv` covers **every one of the 115 outputs** the production
run recorded, with a disposition for each: included directly, included
compressed with row identifiers anonymised, rewritten so its paths resolve in
this package, or held in the external model bundle. Bundle rows carry
`bundle_name` and `bundle_member` and leave `public_file` empty, because those
artifacts are **not inside this package**.

## Privacy of the released predictions

Row identifiers in `frozen_predictions/` are **opaque, release-local tokens**
(`row_000001`, ...). Source row indices are not published, so a released row
cannot be used to index a record in the provider's file. For Diabetes-130 the
cluster identifier is an anonymous token (`P000001`, ...) that preserves patient
grouping exactly, which is all the joint patient bootstrap needs; the original
`patient_nbr` is not published. Outcomes, probabilities, predictions, subgroup
labels, and the clustering structure are exactly what the analysis evaluated.

## Verifying the reference outputs

The run carries two attestations, because 30 files were legitimately amended
after it completed: `reference_outputs/run_manifest.json` attests to the run at
completion, and `reference_outputs/POST_RUN_ARTIFACT_AMENDMENT.json` attests to
the 15 fitted artifacts and 15 companion manifests whose scoring-interface
metadata was corrected afterwards. No model was refitted and no probability
changed.

```
python scripts/verify_run_integrity.py --run-dir <run>
```

expects 85 outputs matching the run manifest, 30 matching the amendment, and 0
unexplained mismatches.

## Fitted model artifacts

The fitted preprocessing, estimator, and calibration objects for all 15
comparisons are serialized and each is verified by reloading it and reproducing
the persisted probabilities. Thirteen are small; two exceed the 100 MB per-file
limit of ordinary Git hosting and are therefore distributed as an archival
bundle rather than in this repository. `model_bundle_manifest.csv` records every artifact's SHA-256, size, required
predictor columns, software versions, and verification outcome. The artifacts
themselves are attached to the v1.1.0 release as
`fitted_models_v1_1_0.tar.gz` (158 MB compressed, 673 MB extracted); they are
**not inside this repository**, because two of them exceed the 100 MB per-file
limit of ordinary Git hosting. This is a distribution
constraint, not a serialization failure.

Artifacts are Python pickles tied to the package versions in
`reference_outputs/run_manifest.json`. The prediction files are the portable,
durable record.

## Citation

See `CITATION.cff`. **No DOI is claimed for this version yet.** The manuscript
is not published or accepted, and this repository makes no such claim.
