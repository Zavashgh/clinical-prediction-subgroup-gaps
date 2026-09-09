# Prevalence, Thresholds, and Subgroup Performance Gaps in Clinical Prediction Models: A Multi-Dataset Analysis

Analysis code and reference outputs for the corrected JAMIA Open submission.

## Project description

This is a retrospective secondary analysis of 5 publicly available clinical
prediction datasets. For each dataset and each demographic attribute, one
prediction model is developed on a training partition and evaluated once on a
held-out partition, and subgroup performance differences are examined. The
primary descriptive endpoint is the **within-dataset sensitivity difference**
between two compared groups. Predictive values are standardized to a common
prevalence, operating-threshold dependence is examined, and case-mix analyses
are reported as exploratory.

Sex and age are evaluated in all 5 datasets; race/ethnicity in 3.

The study is a **methodological evaluation**. The models are comparative
research models for studying subgroup performance across heterogeneous
datasets. They were not developed for clinical deployment, treatment
selection, or individual patient decision support, and no care-pathway role or
intended clinical user is proposed.

## Repository scope

This repository contains **only** the final corrected analysis: the code that
produced the validated corrected production run
`20260909T010240Z_194231425561_909244de` (source commit `1942314255619`),
plus a compact set of reference outputs from that run.

It deliberately does **not** contain manuscript files, manuscript drafts,
earlier or superseded analysis versions, historical production runs, reviewer
material, or the source datasets. It also omits the working repository's
results-promotion module, which is not required to execute the corrected
analysis, regenerate results, or import any analytical module, and which has
no meaning in a public clone. See `PUBLIC_RELEASE_AUDIT.md` for the full
inclusion and exclusion record.

Every file under `src/` is byte-identical to the corresponding file in the
reference run's source commit.

## Corrected methodology summary

Two methodological defects found by independent code audit were corrected, and
this repository contains the corrected implementation only.

**1. All data-derived preprocessing is fitted on training observations only.**
Fixed, deterministic recoding (hard-coded binary and ordinal mappings,
age-band midpoints, a fixed BMI rescaling, medication dose-intensity mappings)
is applied before partitioning, because it consults no observed distribution.
Everything learned from data is fitted after the split:

- median imputation values are computed from training rows only and applied
  unchanged to held-out rows;
- categorical one-hot schemas, including the reference level, are learned from
  training rows only;
- a category appearing only in the held-out partition is mapped to the
  training reference level and never creates a new trained predictor column;
- standardization for the two logistic families happens inside a
  training-fitted scikit-learn `Pipeline`, so scaling parameters are estimated
  within each training fold. Tree-based families are unscaled.

Implemented in `src/preprocessing.py` and applied in `src/pipeline.py`.

**2. Diabetes-130 is partitioned by patient, not by encounter.**
That dataset contains 101,763 encounters from 71,515 unique patients. The
patient identifier is retained as a **grouping variable only** and never
enters the predictor set. Grouping is applied to:

- the outer train/held-out split (`GroupShuffleSplit`, test_size 0.3, seed 42);
- the model-selection cross-validation folds (`StratifiedGroupKFold`, 5 folds);
- the isotonic calibration folds (`StratifiedGroupKFold`, 5 folds);
- the bootstrap, which resamples **patients** within each subgroup and retains
  all of a sampled patient's held-out encounters.

Verified patient overlap between the training and held-out partitions is **0**
in every Diabetes-130 comparison. Because no shuffle-split in scikit-learn
1.9.0 is simultaneously outcome-stratified and grouped, the patient-grouped
outer split is **not** outcome-stratified; realized held-out fractions and
event rates are recorded in the run provenance. The unit of analysis remains
the encounter; no deduplication to one encounter per patient is performed.

The other 4 datasets contain no clustering identifier in their retained
extracts and use outcome-stratified splitting without grouping.

**Model development, unchanged by the correction.** Six fixed model families
(logistic regression, L1-penalized logistic regression, random forest, shallow
decision tree, XGBoost, soft-voting ensemble) with configurations fixed in
advance and **no hyperparameter search**. Family selection uses 5-fold
cross-validated AUROC computed on **training data only**; families with mean
CV AUROC of 0.5 or less are ineligible. The selected family is calibrated with
an explicit 5-fold isotonic fold ensemble on training data. The operating
threshold is the **training-sample outcome prevalence**, applied identically
to both compared groups. **No class weighting, oversampling, undersampling, or
SMOTE is used in any analysis reported in the manuscript.** Held-out outcomes
are not used for model-family selection, fitting, calibration, or primary
threshold choice; the held-out partition is first used for final evaluation.

`tests/test_corrected_partitioning.py` asserts these properties directly,
including an adversarial check that flipping every held-out outcome leaves
model selection, cross-validation scores, calibrated probabilities, and the
threshold unchanged.

## Public dataset sources

**No dataset files are included in this repository.** Obtain each from its
official source. Place the resulting files in `data/` with the filenames the
loaders expect.

| Dataset | Expected file in `data/` | Source |
|---|---|---|
| CDC Diabetes Health Indicators (BRFSS 2015) | `cdc_diabetes.csv` | UCI ML Repository, https://doi.org/10.24432/C53919 |
| Diabetes 130-US Hospitals, 1999-2008 | `diabetes_130_hospitals.csv` | UCI ML Repository, https://doi.org/10.24432/C5230J |
| BRFSS 2022 | `brfss2022_subset.csv` | CDC, https://www.cdc.gov/brfss/annual_data/annual_2022.html — build with `scripts/fetch_brfss2022.py` |
| NHANES 2017-2018 | `nhanes_2017_2018.csv` | NCHS, https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overview.aspx?BeginYear=2017 — build with `scripts/fetch_nhanes.py` |
| CCHS 2019-2020 Annual Component | `cchs_2019_2020_subset.csv` | Statistics Canada via Borealis, https://doi.org/10.5683/SP3/ZVCGBK — build with `scripts/fetch_cchs.py` |

The two UCI datasets are downloaded manually from the links above. The BRFSS,
NHANES, and CCHS helpers build the reduced column subsets the loaders read.
CCHS additionally requires a free Borealis account and acceptance of the
Statistics Canada Open Licence; `scripts/fetch_cchs.py` documents the manual
steps and then extracts the needed columns from the downloaded `.sav` file.

Input file checksums for the reference run are recorded in
`reference_outputs/run_manifest.json` under `inputs`, so you can confirm you
have the same source data.

## Directory structure

```
.
├── README.md
├── LICENSE                       MIT, for the code in this repository
├── CITATION.cff
├── requirements.txt              pinned to the reference run's environment
├── PUBLIC_RELEASE_AUDIT.md       what was included, excluded, and why
├── data/                         empty; you place source datasets here
├── src/                          analysis library (byte-identical to the reference run)
│   ├── datasets.py               dataset loaders and fixed recoding
│   ├── preprocessing.py          training-fitted imputation and one-hot schema
│   ├── pipeline.py               split, selection, calibration, thresholding, evaluation
│   ├── metrics.py                subgroup metrics
│   ├── adjustments.py            prevalence standardization, bootstraps, case-mix
│   ├── extended_analyses.py      threshold sweeps and supplementary analyses
│   ├── evaluation_splits.py      auxiliary holdouts with disjointness assertions
│   ├── figures.py                calibration and gap plots
│   ├── runtime.py                deterministic thread/seed environment
│   ├── output_contracts.py       output schema checks
│   ├── reproducibility_*.py      reproducibility policy and preflight
│   └── production.py             run orchestration and provenance
├── scripts/
│   ├── run_production.py         staged production entry point
│   ├── run_deterministic.py      run one step under the fixed thread profile
│   ├── run_jama_supplement.py    supplementary threshold and case-mix analyses
│   ├── build_jama_reporting_sources.py   table and figure source data
│   └── fetch_*.py                dataset acquisition helpers
├── notebooks/                    the 5 sex analyses executed by the run
├── run_age_decomposition.py      age analyses
├── run_race_decomposition.py     race/ethnicity analyses
├── tests/                        suite run by the reproduction command
│   ├── test_corrected_partitioning.py   corrected-methodology assertions
│   ├── test_jama_scope.py               reporting-contract assertions
│   └── release_suite.py                 public validation command
├── validation_reference/         further tests, run by the validation command
└── reference_outputs/            compact outputs from the reference run
```



## Environment and setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The reference run used Python 3.11.9 on Windows with an 8-thread profile. The
pinned versions in `requirements.txt` are those recorded in the run manifest.

## Two commands: reproduction and validation

This release distinguishes two things deliberately.

### Scientific reproduction

```bash
python scripts/run_production.py plan-jama
python scripts/run_production.py run-jama --execute-production
```

This is the analytical pipeline. `src/production.py` is **byte-identical** to
the version that produced the reference run, so this command runs exactly the
orchestration the manuscript used. It executes, in fixed order: the validation
suite in `tests/`, three preflight checks, the 5 sex notebooks, the age and
race decompositions, the supplementary analyses, and the reporting-source
builder, then re-runs the validation suite. Outputs go to a new timestamped
directory under `production_runs/`. Nothing is promoted or overwritten.

The orchestrator requires a clean tracked git tree and writes a run manifest
with input checksums, environment, per-comparison provenance, and output
verification.

### Public-release validation

```bash
python tests/release_suite.py
```

This is the broader test command for this release. It runs everything in
`tests/` **and** everything in `validation_reference/`, excluding six specific
guards that check for artifacts of the private working repository which are
deliberately not shipped here. Each exclusion is named and justified in
`tests/release_suite.py` and `validation_reference/README.md`. It exits
non-zero on any failure, and also if its exclusion list ever goes stale.

Expect **65 tests, 6 documented exclusions**.

`validation_reference/` is kept outside `tests/` for one reason: so that those
six inapplicable guards cannot abort the reproduction command, and so that no
change to the analytical production code is required to accommodate them.

### Running individual steps

```bash
python scripts/run_deterministic.py -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_cdc_diabetes.ipynb
python scripts/run_deterministic.py run_age_decomposition.py --jama-only
python scripts/run_deterministic.py run_race_decomposition.py --jama-only
python scripts/run_deterministic.py scripts/run_jama_supplement.py
python scripts/run_deterministic.py scripts/build_jama_reporting_sources.py
```

## Expected major outputs

Written under the new run's `staging/results/`:

- `0{1..5}_*_summary.csv` — sex analyses per dataset
- `extended/age_decomposition/age_raw_gaps.csv`, `age_bootstrap.csv`,
  `age_casemix_waterfall.csv`, `age_ipw_adjustment.csv`
- `extended/race_decomposition/race_raw_gaps.csv`, `race_bootstrap.csv`,
  `race_casemix_waterfall.csv`, `race_ipw_adjustment.csv`,
  `race_secondary_comparisons.csv`, `race_stability_checks.csv`
- `extended/jama_supplement/threshold_sweep.csv`, `threshold_policy.csv`,
  `prevalence_standardized_metrics.csv`, `case_mix_point_estimates.csv`
- `jama_reporting/table1_source.csv` — Table 1 data
- `jama_reporting/figure1_source_values.csv` — Figure 1 data
- `jama_reporting/etable1_subgroup_metrics_source.csv` — eTable 1 data
- `jama_reporting/missing_data_flow.csv` — data-flow and predictor missingness

The `jama_reporting/` files are the data behind the manuscript's Table 1,
Figure 1, and eTable 1. Rendering those displays into the manuscript document
is done separately and is not part of this repository.

Compare against `reference_outputs/` to verify.

## Reproducibility expectations

The reference run's own acceptance policy is **tolerance-based, not bitwise**.
Model family, operating threshold, subgroup and event counts, direction and
significance classifications, schemas, and output paths are expected to match
exactly; manuscript metrics and bootstrap intervals are checked to an absolute
tolerance of 1e-12 where deterministic; continuous AUROC and calibrated-score
diagnostics use an absolute tolerance of 1e-5.

**Bitwise identity across different hardware, operating systems, BLAS builds,
or package versions is not claimed and has not been verified.** Two
independent complete runs on the reference machine produced byte-identical
summary outputs, but that is a single-environment observation. Expect small
floating-point differences elsewhere, particularly in XGBoost and threaded
linear algebra.

## Data availability and redistribution

No source dataset is redistributed here. All 5 are publicly obtainable from
the agencies and repositories listed above. Redistribution terms differ by
source, and the CCHS public-use microdata file in particular is distributed
under the Statistics Canada Open Licence through Borealis and requires the
user to obtain it directly. Processed analytic extracts are also not
redistributed; the fetch helpers rebuild them from the official downloads.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The license covers the code only. Source datasets remain under the terms of
their respective providers.

The copyright line in `LICENSE` names the five authors. **Authors should
confirm the correct copyright holder** (the authors individually, or an
institution) before publication; no institutional ownership is claimed here.

## Citation

See `CITATION.cff`. Journal, DOI, and repository identifiers are omitted until
they exist.

## Statement

This repository reproduces the **corrected** JAMIA Open analysis. It contains
the implementation that generated the validated corrected production run
`20260909T010240Z_194231425561_909244de` and no superseded analysis version.
