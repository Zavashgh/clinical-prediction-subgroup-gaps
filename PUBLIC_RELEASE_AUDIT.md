# Public release audit

Audit of `public_release/`, prepared as a self-contained public repository for
the corrected JAMIA Open analysis.

**Reference production run:** `20260909T010240Z_194231425561_909244de`
**Analysis source commit:** `19423142556194cb6545eaf96896e69a8ca0e526`
**Run status:** `complete` · **validation:** `passed` · 53/53 expected outputs

**Final cleanup applied.** Every file under `src/` is now **byte-identical**
to the reference run's source commit, including `src/production.py`. No
analytical or orchestration file is modified in this release. The earlier
adaptation of `src/production.py` has been reverted, and the reduced public
test set is accommodated by test-file placement instead. See §4.

---

## 1. How the file set was selected

Not by filename. The set is the **transitive import closure** of the entry
points the reference run actually executed, taken from that run's own
`run_manifest.json` command plan:

| Command | Entry point |
|---|---|
| `preflight_unit_tests`, `postflight_unit_tests` | `tests/` |
| `preflight_six_model_families` | `src.pipeline._build_models` |
| `preflight_output_contracts` | `src/output_contracts.py` |
| `preflight_eight_thread_reproducibility` | `src/reproducibility_preflight.py` |
| `sex_cdc` … `sex_cchs` | `notebooks/01`–`05` |
| `age_decomposition` | `run_age_decomposition.py` |
| `race_decomposition` | `run_race_decomposition.py` |
| `jama_supplement_analyses` | `scripts/run_jama_supplement.py` |
| `jama_reporting_sources` | `scripts/build_jama_reporting_sources.py` |

The closure was computed with an AST walk over `import` / `from … import`
statements, including relative imports and notebook code cells, resolving each
module against the repository tree. It yielded **25 core files** plus **7 test
files**.

The working tree was verified byte-identical to the run commit for all of
`src/`, `notebooks/`, `tests/`, the two decomposition runners, and the two
scripts (`git diff <run commit> HEAD -- …` returned empty), so the released
code is exactly the code that produced the run.

## 2. Files copied, with original source paths

All paths are relative to the original project root. Copied unmodified unless
noted in §4.

**Analysis library — `src/` (14 files, all byte-identical to the run commit)**
`src/__init__.py`, `src/adjustments.py`, `src/datasets.py`,
`src/evaluation_splits.py`, `src/extended_analyses.py`, `src/figures.py`,
`src/metrics.py`, `src/output_contracts.py`, `src/pipeline.py`,
`src/preprocessing.py`, `src/production.py`,
`src/reproducibility_policy.py`, `src/reproducibility_preflight.py`,
`src/runtime.py`

`src/promotion.py` was **removed** in the final cleanup; see §3a.

**Executed entry points (5)**
`run_age_decomposition.py`, `run_race_decomposition.py`,
`scripts/run_production.py`, `scripts/run_jama_supplement.py`,
`scripts/build_jama_reporting_sources.py`

**Helpers (4)**
`scripts/run_deterministic.py` (documented single-step runner),
`scripts/fetch_brfss2022.py`, `scripts/fetch_nhanes.py`, `scripts/fetch_cchs.py`

**Notebooks (5)**
`notebooks/01_cdc_diabetes.ipynb` … `notebooks/05_cchs.ipynb`

**Validation suite (6 files, unmodified, in two locations)**

`tests/` — run by the reproduction command's pre/postflight:
`test_corrected_partitioning.py`, `test_jama_scope.py`

`validation_reference/` — run by the public validation command:
`test_integrity_protocol.py`, `test_output_contracts.py`,
`test_production_orchestrator.py`, `test_reproducibility_policy.py`

`tests/test_promotion_hardening.py` was **removed** with `src/promotion.py`;
see §3a.

**Reference outputs (25)**
23 result CSVs from
`production_runs/20260909T010240Z_194231425561_909244de/staging/results/`,
plus that run's `run_manifest.json` and `provenance/provenance.jsonl`.

**Files created for this release (9)**
`README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `requirements.txt`,
`data/README.md`, `reference_outputs/README.md`,
`validation_reference/README.md`, `tests/release_suite.py`, and this audit.

## 3. Files intentionally excluded, and why

| Excluded | Reason |
|---|---|
| `data/*.csv` (all 5 source datasets) | Redistribution not established for all sources; CCHS in particular is distributed under the Statistics Canada Open Licence via Borealis and must be obtained by the user. See §7. |
| `extended_robustness_scripts/` | Optional robustness analyses. Not executed by the corrected production plan; their outputs are not current evidence and live in a superseded archive. Excluding them is the reason for 5 of the 6 test exclusions in §4. |
| `notebooks/06_pooled_analysis.ipynb` | Pooled meta-analysis. Not in the corrected plan; pooled estimates are explicitly not current evidence. |
| `production_runs/` (5 historical runs incl. 3 failed/incomplete) | Superseded and bulky. Only compact outputs from the one validated run are included, under `reference_outputs/`. |
| `results/` (promoted live tree) | Belongs to the working repository's promotion workflow. |
| `manuscript/` in its entirety | Manuscript files, drafts, review packages, changelogs, audits, and the JAMIA/JAMA display builders. Includes unpublished manuscript prose. |
| `archive/`, `audit/`, `.promotion_backups/`, `jama_editorial/`, `tmp/` | Historical artifacts, reviewer material, backups, scratch. |
| `citations/`, `references/` | Third-party PDFs (journal articles, TRIPOD+AI checklists). Redistribution not permitted. |
| `environment/requirements-host-2026-07-20.txt` | Full host-machine environment snapshot containing many unrelated packages. Replaced with a minimal pinned `requirements.txt` derived from the run manifest. |
| `build_jamia_docx.py`, `.audit_runtime.py` | Legacy and scratch tooling not in the corrected plan. |
| `CLAUDE.md`, `AGENTS.md`, `HANDOFF_INTERNAL.md`, `README.md` (original) | Internal working instructions and handoff notes. |
| `__pycache__/`, `.claude/`, `.agents/` | Caches and local tooling configuration. |
| `*.docx`, `*.pdf` anywhere | Manuscript and reference documents. |

## 3a. Removal of `src/promotion.py`

`src/promotion.py` implemented the working repository's guarded
results-promotion transaction: replacing the live `results/` tree from a
validated run, with backups and recovery. It was **removed** from this release
after verifying it is not required to:

- execute the corrected analysis — all 13 remaining `src` modules import
  without it, and `python scripts/run_production.py plan-jama` builds the full
  14-command plan;
- regenerate manuscript results — no analysis command touches it;
- import any analytical module needed for reproduction — `src/production.py`
  imports it **lazily, inside four wrapper functions** (`build_promotion_dry_run`,
  `verify_promotion_backup_dry_run`, `recover_interrupted_promotion`,
  `promote_results`), never at module scope.

Those four wrappers are reachable only through the `promote` and
`recover-promotion` subcommands of `scripts/run_production.py`, which have no
meaning in a public clone: there is no live `results/` tree to promote into.
Invoking them in this release raises `ModuleNotFoundError`; the reproduction
path never does.

**Removing it required no change to the corrected analytical implementation.**
`src/production.py` remains byte-identical to the run commit. Its test file,
`tests/test_promotion_hardening.py` (25 tests, entirely promotion
infrastructure), was removed with it.

## 4. Code adapted for the release

**No code was adapted. Every released file is byte-identical to its source.**

An earlier draft of this release modified `src/production.py` so that its pre-
and postflight steps ran a filtered test suite. That change has been
**reverted**; `src/production.py` is now byte-identical to the reference run's
source commit (SHA-256 `f18f117b6ec628c2…`, matching both the run commit and
the working tree), with zero adaptation markers.

The problem it was working around is instead solved by **test-file
placement**, which requires no code change.

### The problem

Six tests in the working repository's suite are guards over artifacts of that
private repository which are deliberately not shipped here:

| Test | Requires |
|---|---|
| `test_integrity_protocol.test_notebook_production_sources_use_rooted_paths_and_primary_threshold` | exactly 6 notebooks, including the excluded pooled meta-analysis |
| `test_integrity_protocol.test_robustness_baselines_do_not_restore_cdc_fixed_threshold` | `extended_robustness_scripts/*.py` |
| `test_output_contracts.test_actual_production_plot_consumer_completes` | `extended_robustness_scripts/run_extended_analyses.py` |
| `test_output_contracts.test_calibration_summary_has_flat_unique_round_trip_schema` | `extended_robustness_scripts/run_extended_13_20_34.py` |
| `test_production_orchestrator.test_every_direct_production_runner_accepts_staging_results` | more than 2 direct runners, counting `extended_robustness_scripts/` |
| `test_reproducibility_policy.test_focused_cchs_evidence_satisfies_conditional_acceptance` | a dated diagnostic JSON from the working repository's `audit/` directory |

`src/production.py` runs `python -m unittest discover -s tests` at pre- and
postflight, and `execute_production_plan` returns immediately on the first
failed command. So if any of these lived under `tests/`, the reproduction
command would abort on a clean clone.

### The solution

The four affected files are placed in **`validation_reference/`** instead of
`tests/`. They are unmodified and fully readable.

- `tests/` contains only files in which every test passes in this release:
  `test_corrected_partitioning.py` and `test_jama_scope.py`. So
  `unittest discover -s tests` passes and the reproduction command works with
  the unmodified orchestrator. **23 tests.**
- `tests/release_suite.py` is the separate, documented public validation
  command. It runs `tests/` **and** `validation_reference/`, excluding only
  the six named guards, and fails if that exclusion list ever goes stale.
  **65 tests, 6 exclusions.**

`release_suite.py` does not match the `test*.py` discovery pattern, so it is
not picked up by the reproduction command's preflight.

Every excluded item is a repository-shape, historical-artifact, or
promotion-infrastructure guard. **None validates the corrected preprocessing,
partitioning, calibration, thresholding, or evaluation.** The alternative —
shipping the excluded scripts so the guards pass — was rejected because it
would reintroduce superseded analysis versions.

## 5. Dependency check

The release was copied to an isolated directory outside the project and
exercised there:

- all 13 `src` modules import standalone with `promotion.py` absent
  (`src.reproducibility_preflight` requires `MEDICAL_FAIRNESS_N_JOBS=8`, which
  is its designed guard and is set by the orchestrator);
- `python scripts/run_production.py plan-jama` prints the full 14-step plan;
- `python -m unittest discover -s tests`, which is what the reproduction
  command runs at pre- and postflight → **23 tests, OK**;
- `python tests/release_suite.py` → **65 tests selected, 6 excluded, OK**,
  exit code 0;
- `src.datasets` resolves data paths to the **release's own** `data/`
  directory, confirmed by the `FileNotFoundError` path on a clean copy.

**No dependency on any file outside the release remains**, other than the
user-supplied source datasets.

Third-party runtime dependencies, pinned in `requirements.txt` from the run
manifest: matplotlib, numpy, pandas, scipy, scikit-learn, statsmodels,
xgboost, jupyter, nbconvert, pyreadstat, threadpoolctl.

## 6. Secret and privacy check

Swept every `.py`, `.ipynb`, `.md`, `.cff`, and `.txt` in the release for API
keys, tokens, passwords, credentials, private keys, absolute paths, home
directories, and email addresses.

- **No secrets, tokens, credentials, or private keys.**
- **No absolute or machine-local paths.** All roots derive from `__file__`.
  The only `C:\` string is a documentation placeholder in
  `scripts/fetch_cchs.py` showing the argument format.
- **Path redaction applied to reference provenance.** A pre-push check found
  that `reference_outputs/run_manifest.json` and
  `reference_outputs/provenance.jsonl` recorded the absolute interpreter path
  and project-root path from the machine that produced the run, exposing a
  local user name and directory layout. 75 path strings were redacted
  (`python_executable` and `argv` entries only). Every other value was
  verified unchanged: 10,457 non-path scalar leaves are identical, including
  all input checksums, protocol fields, cross-validation scores, partition
  counts, and metrics. See `reference_outputs/README.md`.
- **Stale notebook outputs removed.** A pre-push check found the five source
  notebooks carried 106 stored outputs, including 25 embedded PNGs, produced
  by a run that predates the methodological correction: the Diabetes-130
  figures (train n=71,234, test n=30,529, sensitivity gap -0.0596) match
  neither the corrected run nor the superseded promoted run, and the outputs
  lack the `Split protocol:` and `Clusters:` lines the corrected pipeline
  prints. Publishing them would have rendered pre-correction numbers inline.
  Outputs and execution counts were cleared; every code cell is
  byte-identical, verified programmatically.
- **No personal information.** The only email-shaped string is
  `synthetic@example.invalid`, a placeholder in one retained test fixture
  (`validation_reference/test_production_orchestrator.py`).
- Author names appear only in `LICENSE` and `CITATION.cff`, as intended.
- No institutional email, phone number, or postal address is present. (Those
  appear only in the manuscript, which is excluded.)

## 6a. Pre-publication sanitation and history rewrite

Three sanitation actions were applied to the release before it was made
available for inspection. **None changed any scientific result, any analytical
code, or any methodological provenance.**

### Stale notebook outputs removed

The five source notebooks carried **106 stored outputs**, including 25
embedded base64 PNGs, produced by a run that predates the methodological
correction. The stored Diabetes-130 figures (train n=71,234, test n=30,529,
sensitivity gap -0.0596) match neither the corrected run (-0.048572) nor the
superseded promoted run (-0.061615), and the outputs contain no
`Split protocol:` or `Clusters:` lines, which the corrected pipeline prints.
They were produced by the uncorrected encounter-level-split implementation,
and publishing them would have rendered pre-correction numbers inline.

Outputs and execution counts were cleared. **Every code cell is
byte-identical**, verified programmatically cell by cell. No analytical code
was modified.

### Machine-local provenance paths sanitized

`reference_outputs/run_manifest.json` and `reference_outputs/provenance.jsonl`
recorded absolute paths from the machine that produced the run, exposing a
local operating-system user name, the home-directory layout, and the local
Python installation path.

Sanitized in two passes, limited to path-valued fields:

| Field | Before | After |
|---|---|---|
| `environment.python_executable` | absolute interpreter path under the user's home directory | `python` |
| `commands[].argv[]` / `argv[]` (interpreter token) | same absolute interpreter path | `python` |
| `commands[].argv[]` / `argv[]` (notebook paths) | absolute path rooted at the local project directory | repository-relative, e.g. `production_runs/<run_id>/prepared_notebooks/01_cdc_diabetes.ipynb` |

**95 path strings were replaced in total** (75 in the first pass, 20 in the
second normalization to repository-relative form).

Verified by comparing every scalar leaf before and after each pass: all
non-path values are identical. That includes every input SHA-256 checksum,
run identifier, git commit, random seed, protocol field, candidate-family
cross-validation score, partition size and event count, operating threshold,
calibration configuration, and reported metric.

Fields deliberately **retained** because they are portable and carry genuine
provenance value: `environment.python_version`, `environment.packages`,
`git.commit`, `git.branch`, `inputs[].path` and `inputs[].sha256`,
`outputs[].path` and their hashes, and the full thread-environment record.
All were already repository-relative or machine-independent.

**No scientific provenance was lost.** The only information removed is where
the interpreter lived on one particular computer.

### Git history rewritten before public release

The first push to the private repository contained the unsanitized paths in
its initial commit. Removing them in a later commit would have left the values
recoverable in history, so the public repository's history was **rewritten**:
the previous commits were discarded and replaced with a single clean initial
commit built from the sanitized tree, then force-pushed.

The messy parent working repository's history was **not** altered.

**Outcome.** The repository that carried the unsanitized initial commit was
deleted, and the repository was recreated under the correct account from the
sanitized tree. It therefore has no prior history at all: a single initial
commit, verified by scanning every object in the repository for the local user
name, Windows and Unix home paths, `AppData`, and `site-packages`, with zero
matches.

**One item is recorded rather than removed.** The single commit's author
metadata carries the corresponding author's institutional email address. This
is ordinary git attribution, is the same address published as the
corresponding-author contact, and is not a machine-local path or a secret. It
is noted here so the authors can decide whether to rewrite it before public
release.

## 7. Dataset redistribution check

**No dataset file is redistributed.** `data/` ships empty apart from a
`.gitkeep` and a README, and `.gitignore` excludes its contents.

| Dataset | Terms | Handling |
|---|---|---|
| CDC Diabetes Health Indicators | UCI ML Repository | Linked by DOI; user downloads |
| Diabetes 130-US Hospitals | UCI ML Repository | Linked by DOI; user downloads |
| BRFSS 2022 | US federal public data | `scripts/fetch_brfss2022.py` downloads and subsets |
| NHANES 2017-2018 | US federal public data | `scripts/fetch_nhanes.py` downloads and subsets |
| CCHS 2019-2020 PUMF | Statistics Canada Open Licence, via Borealis; requires an account and licence acceptance | **Cannot be redistributed or auto-downloaded.** `scripts/fetch_cchs.py` documents the manual steps and then extracts the needed columns locally |

Processed analytic extracts are also not redistributed; the fetch helpers
rebuild them from the official downloads. Input SHA-256 checksums in
`reference_outputs/run_manifest.json` let a user confirm they obtained the
same inputs.

## 8. License check

All released code is first-party, authored within this project. No third-party
source is vendored: every external dependency is imported from PyPI at runtime
and pinned in `requirements.txt` rather than copied in.

Runtime dependencies are BSD-3-Clause (numpy, pandas, scipy, scikit-learn,
statsmodels), Apache-2.0 (xgboost), PSF/matplotlib, and BSD-3-Clause
(jupyter, nbconvert, threadpoolctl); `pyreadstat` is Apache-2.0 and links
ReadStat (MIT). None of these imposes a copyleft obligation on code that
merely imports them.

**MIT is appropriate and there is no license conflict.** The `LICENSE` file
covers the code only; the datasets remain under their providers' terms, as
stated in the README.

**Copyright holder requires author confirmation.** The `LICENSE` copyright
line currently names the five authors. No institutional ownership is claimed
and none should be inferred. If the work is institutionally owned, the authors
must correct this line before publication. This audit does not guess.

## 9. Reproducibility limitations

- **Bitwise reproducibility is not claimed** across differing hardware,
  operating systems, BLAS builds, or package versions, and has not been
  verified outside the reference machine. Two independent complete runs on
  that machine produced byte-identical summary outputs; that is a
  single-environment observation.
- The run's own acceptance policy is tolerance-based: exact agreement is
  required for model family, threshold, counts, direction and significance,
  schemas, and output paths; 1e-12 for manuscript metrics and bootstrap
  intervals where deterministic; 1e-5 for continuous AUROC and calibrated
  score diagnostics.
- The orchestrator requires a clean tracked git tree, so a clone must be
  committed-clean before `run-jama`.
- Reproduction requires all 5 datasets, one of which (CCHS) needs a Borealis
  account and manual download.
- The Diabetes-130 patient-grouped outer split is **not** outcome-stratified,
  because scikit-learn 1.9.0 offers no simultaneously stratified and grouped
  shuffle-split. Realized held-out fractions and event rates are recorded in
  the run provenance.
- Fitted preprocessing, model, and calibration objects are **not** serialized.
  The analysis is reproducible by retraining from the original data sources;
  the exact fitted estimators cannot be loaded.

## 10. Corrected methodology verification

Re-verified after the final cleanup, in a fresh isolated copy. All 15
properties present:

| Property | Where |
|---|---|
| Training-only median imputation | `src/preprocessing.py` `TrainFittedPreprocessor.fit` |
| Loaders no longer impute or one-hot before the split | `src/datasets.py` (no `get_dummies`, no median `fillna`) |
| Training-only categorical schema and reference level | `src/preprocessing.py` `reference_level_` |
| Held-out-only category maps to reference, creates no column | `src/preprocessing.py` `unseen_category_counts_` |
| Diabetes-130 patient-grouped split | `src/pipeline.py` `GroupShuffleSplit` |
| Zero patient overlap enforced, raises otherwise | `src/pipeline.py` "Patient-grouped split failed" |
| Patient-grouped model-selection CV | `src/pipeline.py` `StratifiedGroupKFold` + `selection_fit_params` |
| Patient-grouped calibration folds | `src/pipeline.py` `GROUPED_CALIBRATION_PROTOCOL`, `_assert_no_cluster_leakage_in_folds` |
| Patient-cluster bootstrap | `src/adjustments.py` `bootstrap_gap_ci_clustered` |
| No class weighting, oversampling, undersampling, or SMOTE | absent from `src/pipeline.py` and `src/datasets.py` |
| Training-only model-family selection | `src/pipeline.py` `cross_val_score(..., X_train, y_train)` |
| Isotonic calibration | `src/pipeline.py` `method="isotonic"` |
| Training-prevalence threshold | `src/pipeline.py` `float(y_train.mean())` |
| Held-out outcomes excluded from development | `src/pipeline.py` single final evaluation |
| `patient_nbr` never a predictor | `src/datasets.py` `cluster_col = "patient_nbr"` |

`tests/test_corrected_partitioning.py` asserts these behaviourally, including
an adversarial test that flipping every held-out outcome leaves model
selection, CV scores, calibrated probabilities, and the threshold unchanged.
All 13 of its tests pass in the isolated copy, within the 23 that the
reproduction command runs and the 65 that the validation command runs.

## 11. Confirmation: nothing outside the release was modified

Verified by `git status`:

- `results/` — 0 changes, 57 files, untouched.
- `production_runs/20260909T010240Z_194231425561_909244de/` — 0 changes.
- `manuscript/` — 0 changes.
- `src/`, `scripts/`, `notebooks/`, `tests/` at the project root — 0 changes.

`public_release/` was created by copying only. Nothing in the working project
was deleted, renamed, reorganized, or edited. The only file modified inside
the release is the release's own copy of `src/production.py` (§4).
