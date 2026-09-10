# data/

**This directory is intentionally empty.** No source dataset is redistributed
with this repository.

Download each dataset from its official source and place it here using the
filename the loaders expect:

| Expected filename | Dataset | Source |
|---|---|---|
| `cdc_diabetes.csv` | CDC Diabetes Health Indicators (BRFSS 2015) | UCI ML Repository, https://doi.org/10.24432/C53919 |
| `diabetes_130_hospitals.csv` | Diabetes 130-US Hospitals, 1999-2008 | UCI ML Repository, https://doi.org/10.24432/C5230J |
| `brfss2022_subset.csv` | BRFSS 2022 (column subset) | build with `python scripts/fetch_brfss2022.py` |
| `nhanes_2017_2018.csv` | NHANES 2017-2018 (column subset) | build with `python scripts/fetch_nhanes.py` |
| `cchs_2019_2020_subset.csv` | CCHS 2019-2020 Annual Component (column subset) | build with `python scripts/fetch_cchs.py <path to cchs_201920_pumf.sav>` |

The two UCI files are downloaded directly from the DOI links.

The BRFSS and NHANES helpers download from the CDC/NCHS and write the reduced
column subsets used by `src/datasets.py`.

CCHS requires a free Borealis account and one-click acceptance of the
Statistics Canada Open Licence; it cannot be downloaded programmatically.
`scripts/fetch_cchs.py` documents the manual steps and then extracts the
needed columns from the downloaded SPSS file.

To confirm you have the same inputs as the reference run, compare SHA-256
checksums against the `inputs` block of
`reference_outputs/run_manifest.json`.

Files placed here are ignored by git (see `.gitignore`).
