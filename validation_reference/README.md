# validation_reference/

Test files retained from the working repository for reference and for the
public validation suite, but kept **outside** `tests/`.

## Why they are here rather than in `tests/`

`src/production.py` in this release is byte-identical to the version that
produced the reference run. Its pre- and postflight steps run
`python -m unittest discover -s tests`, and the orchestrator aborts the whole
run on the first failing command.

Each file here contains substantial methodological coverage **and** one or two
guards over artifacts of the private working repository that are deliberately
not shipped in this release. Left in `tests/`, those guards would fail on a
clean clone and abort the scientific reproduction command — even though
nothing about the analysis is wrong.

Keeping them here means:

- the analytical production code stays unmodified;
- `python scripts/run_production.py run-jama --execute-production` works on a
  clean clone;
- none of this coverage is lost, because
  `python tests/release_suite.py` runs these files too, with only the specific
  inapplicable guards excluded.

## What is excluded when the release suite runs these files

| File | Excluded test | Requires |
|---|---|---|
| `test_integrity_protocol.py` | `test_notebook_production_sources_use_rooted_paths_and_primary_threshold` | exactly 6 notebooks, including the excluded pooled meta-analysis |
| `test_integrity_protocol.py` | `test_robustness_baselines_do_not_restore_cdc_fixed_threshold` | `extended_robustness_scripts/*.py` |
| `test_output_contracts.py` | `test_actual_production_plot_consumer_completes` | `extended_robustness_scripts/run_extended_analyses.py` |
| `test_output_contracts.py` | `test_calibration_summary_has_flat_unique_round_trip_schema` | `extended_robustness_scripts/run_extended_13_20_34.py` |
| `test_production_orchestrator.py` | `test_every_direct_production_runner_accepts_staging_results` | more than 2 direct runners, counting `extended_robustness_scripts/` |
| `test_reproducibility_policy.py` | `test_focused_cchs_evidence_satisfies_conditional_acceptance` | a dated diagnostic JSON from the working repository's `audit/` directory |

Every excluded item is a repository-shape or historical-artifact guard. None
of them validates the corrected preprocessing, partitioning, calibration,
thresholding, or evaluation.

## What these files still validate

Everything else in them runs, including training-only family selection and
calibration, the six-family candidate pool, the training-prevalence threshold
rule, the explicit isotonic fold-ensemble calibration protocol, deterministic
family selection under the fixed thread profile, working-directory-independent
dataset paths, auxiliary-holdout training isolation, strict-JSON provenance
capture, output schema contracts, orchestrator staging and provenance
behaviour, and the reproducibility tolerance policy.

The files are unmodified.
