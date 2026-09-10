# Post-run artifact amendment

**This is a second attestation, not a replacement for the run manifest.**

| | |
|---|---|
| Original run | `20260909T234626Z_5b0145ea3bb0_e89e8377` |
| Analysis source commit | `5b0145ea3bb0801551f5f343c24b9366b3192a1d` |
| Run completed | 2026-09-10T00:07:38.021229+00:00 |
| Repack commit | `f18f63a` |
| Repack committed | 2026-09-09T21:18:56-04:00 |
| Amendment written | 2026-09-10T04:43:54.469182+00:00 |

## What changed and why

Scoring-interface metadata correction only. The artifacts advertised every column of the working frame as a required scoring input, including the outcome, the demographic subgroup label, and a source identifier. They now declare only the raw predictor columns the model consumes.

**No estimator, preprocessor, or calibration object was refitted, and no
predicted probability changed.** Only the metadata describing what a caller must
supply in order to score was corrected. Repacking rewrote the files, so their
checksums no longer match the ones recorded at run completion.

## How to read the two records

The run manifest attests to the production run **as it completed**. It has not
been edited, and it must not be read as attesting to files created afterwards.

| Record | Scope |
|---|---|
| `run_manifest.json` | the run's numerical outputs — **85 of 115** recorded outputs still match it byte for byte |
| `POST_RUN_ARTIFACT_AMENDMENT.json` | the 30 metadata-repacked files, and nothing else |

Verify both together with:

```
python scripts/verify_run_integrity.py --run-dir 20260909T234626Z_5b0145ea3bb0_e89e8377
```

which checks unchanged outputs against the manifest, the amended files against
this amendment, and fails on any mismatch that neither record explains.

## Amended files

15 fitted model artifacts and 15 companion frozen
manifests, 30 files in total.

Each artifact below was **independently rescored**: reloaded, used to score the
held-out predictor rows rebuilt from its source dataset, and compared against
the frozen probabilities recorded in `results/frozen/<slug>_predictions.csv`.

| Comparison | Family | Original SHA-256 | Amended SHA-256 | MB | Rows rescored | Max abs deviation | Status |
|---|---|---|---|---|---|---|---|
| `age_brfss_2022_heart_disease` | xgboost | `d4f84e4e357b0fd8...` | `498a6726f4db37f7...` | 3.1 | 86,744 | 1.1e-16 | verified |
| `age_cchs_2019_20_diabetes` | l1_logistic_regression | `e0f58c1195cc3304...` | `263b601baa41ef8c...` | 0.0 | 18,665 | 9.9e-17 | verified |
| `age_cdc_diabetes_brfss_2015` | xgboost | `1d6940411eb4dc25...` | `e5d0563e43c1f62b...` | 3.1 | 43,042 | 9.9e-17 | verified |
| `age_diabetes_130_readmission` | calibrated_ensemble | `d3f1b9b3b1568431...` | `f03d4e2f86d3be3c...` | 385.6 | 21,086 | 4.0e-15 | verified |
| `age_nhanes_2017_18_diabetes` | logistic_regression | `ee6e180ae658f0f7...` | `49abbf76c9c1c220...` | 0.0 | 1,164 | 1.1e-16 | verified |
| `race_brfss_2022_heart_disease_hispanic_vs_white` | xgboost | `4b89f62d4a9de937...` | `290cbcc27ba92ffb...` | 3.1 | 107,918 | 1.1e-16 | verified |
| `race_brfss_2022_heart_disease` | xgboost | `2cf3ec84631f9f02...` | `902c9dd2e2344c0f...` | 3.1 | 105,687 | 1.0e-16 | verified |
| `race_brfss_2022_heart_disease_multiracial_vs_white` | xgboost | `92265b2dbeae955f...` | `891ed7b22708c434...` | 3.1 | 98,038 | 1.0e-16 | verified |
| `race_diabetes_130_readmission` | xgboost | `5efbaa3ba94755b6...` | `3a28f203be83b5e1...` | 5.6 | 28,837 | 1.1e-16 | verified |
| `race_nhanes_2017_18_diabetes` | l1_logistic_regression | `8c38be9fe1063017...` | `6ff8c6c458b8949b...` | 0.0 | 1,013 | 9.7e-17 | verified |
| `sex_brfss` | xgboost | `47ca0cb097d7eb52...` | `369cee43aea6a04f...` | 3.1 | 132,034 | 9.9e-17 | verified |
| `sex_cchs` | calibrated_ensemble | `dca21dd471cdf80e...` | `86671e64404ce023...` | 254.4 | 29,746 | 1.2e-15 | verified |
| `sex_cdc` | xgboost | `805e60a63851aa9a...` | `4ec57ab9ceb37c92...` | 3.1 | 76,104 | 1.0e-16 | verified |
| `sex_diabetes130` | xgboost | `eb5f981534b4f20c...` | `2ff83c731dbf1b7e...` | 5.8 | 30,180 | 9.7e-17 | verified |
| `sex_nhanes` | l1_logistic_regression | `f6646a85aa095170...` | `944765832ae17b51...` | 0.0 | 1,756 | 9.7e-17 | verified |

Maximum deviation across all 15 artifacts:
**3.99e-15** (tolerance 1e-12).

The 15 companion `*_manifest.json` files were rewritten in the same
operation to record the corrected required-input list and the new checksums.
Their original and amended checksums are listed in the JSON record.

## Environment

Python 3.11.9 on Windows-10-10.0.26200-SP0; package versions as
recorded in the original run manifest.
