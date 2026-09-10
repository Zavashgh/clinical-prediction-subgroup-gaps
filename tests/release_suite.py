"""Public-release validation command.

This is the validation entry point for the public release. It is deliberately
separate from the scientific reproduction command, and it does **not** require
any change to the analytical production code:

  Scientific reproduction   python scripts/run_production.py run-jama --execute-production
  Public validation         python tests/release_suite.py

`src/production.py` in this release is byte-identical to the version that
produced the reference run. Its pre- and postflight steps run
`python -m unittest discover -s tests`, which finds only the test files in
`tests/`. Every test in `tests/` passes in this release, so the reproduction
command works unmodified.

Four further test files from the working repository are retained under
`validation_reference/`. They contain substantial methodological coverage, but
each also contains one or two guards over artifacts of the private working
repository that are deliberately not shipped here. They are kept out of
`tests/` so that they cannot break the reproduction command, and this suite
runs them with those specific guards excluded.

Test files retained in `validation_reference/`, and the guards excluded:

  test_integrity_protocol.py
    .test_notebook_production_sources_use_rooted_paths_and_primary_threshold
        Asserts exactly 6 notebooks. This release ships the 5 the corrected
        production run executes; the sixth is a pooled meta-analysis that is
        not part of the corrected analysis and whose results are not current
        evidence.
    .test_robustness_baselines_do_not_restore_cdc_fixed_threshold
        Scans extended_robustness_scripts/, optional robustness scripts the
        corrected plan does not execute and whose outputs are not current
        evidence.

  test_output_contracts.py
    .test_actual_production_plot_consumer_completes
    .test_calibration_summary_has_flat_unique_round_trip_schema
        Both read specific files from extended_robustness_scripts/.

  test_production_orchestrator.py
    .test_every_direct_production_runner_accepts_staging_results
        Requires more than two direct runners, counting the
        extended_robustness_scripts/ runners. This release ships the two the
        corrected plan executes.

  test_reproducibility_policy.py
    .test_focused_cchs_evidence_satisfies_conditional_acceptance
        Reads a dated diagnostic JSON from the working repository's audit/
        directory, a historical artifact of that repository.

One test file from the working repository is **not** included at all:

  test_promotion_hardening.py
        Tests src/promotion.py, the working repository's guarded
        results-promotion transaction. That module is not required to execute
        the corrected analysis, regenerate manuscript results, or import any
        analytical module, and it has no meaning in a public clone, so both it
        and its test file are omitted. See PUBLIC_RELEASE_AUDIT.md.

Every exclusion above is a repository-shape, historical-artifact, or
promotion-infrastructure guard. None validates the corrected preprocessing,
partitioning, calibration, thresholding, or evaluation.

Exits non-zero if any selected test fails, or if the exclusion list has gone
stale relative to what is actually present.
"""

import pathlib
import sys
import unittest

EXCLUDED = {
    ("test_integrity_protocol",
     "test_notebook_production_sources_use_rooted_paths_and_primary_threshold"),
    ("test_integrity_protocol",
     "test_robustness_baselines_do_not_restore_cdc_fixed_threshold"),
    ("test_output_contracts",
     "test_actual_production_plot_consumer_completes"),
    ("test_output_contracts",
     "test_calibration_summary_has_flat_unique_round_trip_schema"),
    ("test_production_orchestrator",
     "test_every_direct_production_runner_accepts_staging_results"),
    ("test_reproducibility_policy",
     "test_focused_cchs_evidence_satisfies_conditional_acceptance"),
}

TESTS_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
REFERENCE_DIR = ROOT / "validation_reference"


def _is_excluded(test):
    module = type(test).__module__.rsplit(".", 1)[-1]
    return (module, test._testMethodName) in EXCLUDED


def _filter(suite, kept, dropped):
    out = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            out.addTest(_filter(item, kept, dropped))
        elif _is_excluded(item):
            dropped.append(item.id())
        else:
            kept.append(item.id())
            out.addTest(item)
    return out


def build_suite():
    """Discover tests/ and validation_reference/, minus the documented guards."""
    sys.path.insert(0, str(ROOT))
    loader = unittest.defaultTestLoader
    combined = unittest.TestSuite()
    kept, dropped = [], []

    for start in (TESTS_DIR, REFERENCE_DIR):
        if not start.is_dir():
            continue
        sys.path.insert(0, str(start))
        discovered = loader.discover(start_dir=str(start), top_level_dir=str(start))
        combined.addTest(_filter(discovered, kept, dropped))

    return combined, kept, dropped


def main():
    suite, kept, dropped = build_suite()
    print(f"public release validation suite")
    print(f"  selected : {len(kept)} tests")
    print(f"  excluded : {len(dropped)} private-repository guards")
    for test_id in sorted(dropped):
        print(f"             {test_id}")
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if len(dropped) != len(EXCLUDED):
        print(
            f"\nERROR: expected to exclude {len(EXCLUDED)} guards, excluded "
            f"{len(dropped)}. The exclusion list in this file is stale "
            f"relative to validation_reference/.",
            file=sys.stderr,
        )
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
