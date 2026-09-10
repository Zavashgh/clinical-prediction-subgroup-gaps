"""Explicit staged production entry point.

This script never promotes results during ``run``.  Promotion is a separate,
guarded subcommand and additionally requires an explicit execution flag.
"""

import argparse
import os
from pathlib import Path
import sys

# These limits are set before importing the orchestration module.  That module
# is standard-library-only, and numerical packages are imported only later by
# deterministic child processes.
_requested_action = sys.argv[1] if len(sys.argv) > 1 else ""
_thread_count = "8" if _requested_action in {"plan-jama", "run-jama"} else "1"
os.environ["MEDICAL_FAIRNESS_N_JOBS"] = _thread_count
_THREAD_ENVIRONMENT = {
    "BLAS_NUM_THREADS": _thread_count,
    "OMP_NUM_THREADS": _thread_count,
    "MKL_NUM_THREADS": _thread_count,
    "OPENBLAS_NUM_THREADS": _thread_count,
    "NUMEXPR_NUM_THREADS": _thread_count,
    "BLIS_NUM_THREADS": _thread_count,
    "VECLIB_MAXIMUM_THREADS": _thread_count,
}
for _name, _value in _THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value
os.environ["PYTHONHASHSEED"] = "0"


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.production import (  # noqa: E402 - deterministic environment first
    ProductionError,
    build_promotion_dry_run,
    create_run_layout,
    execute_production_plan,
    hash_inputs,
    jama_command_specs,
    production_command_specs,
    promote_results,
    recover_interrupted_promotion,
    require_clean_tracked_tree,
    verify_promotion_backup_dry_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("plan", help="Print the fixed production order only")
    subparsers.add_parser(
        "plan-jama", help="Print the manuscript-scoped JAMA production order"
    )
    run = subparsers.add_parser("run", help="Execute into a new staging directory")
    run.add_argument(
        "--execute-production",
        action="store_true",
        help="Required safety acknowledgement; without it no run starts",
    )
    run_jama = subparsers.add_parser(
        "run-jama", help="Execute the manuscript-scoped JAMA plan into staging"
    )
    run_jama.add_argument(
        "--execute-production",
        action="store_true",
        help="Required safety acknowledgement; without it no run starts",
    )
    promote = subparsers.add_parser(
        "promote", help="Replace live results from a validated completed run"
    )
    promote.add_argument("--run-dir", type=Path, required=True)
    promote.add_argument(
        "--disposition-plan",
        type=Path,
        required=True,
        help="Tracked exhaustive live/staged file-disposition plan",
    )
    promote.add_argument(
        "--expected-promotion-code-commit",
        required=True,
        help="Approved full commit hash containing the promotion implementation",
    )
    promote.add_argument(
        "--execute-promotion",
        action="store_true",
        help="Required safety acknowledgement; without it nothing is promoted",
    )
    promote.add_argument(
        "--verify-backup-filesystem",
        action="store_true",
        help=(
            "Create, verify, and remove an ephemeral full backup; never build "
            "a candidate or swap live results"
        ),
    )
    recover = subparsers.add_parser(
        "recover-promotion",
        help="Inspect or explicitly recover an interrupted promotion transaction",
    )
    recover.add_argument(
        "--execute-recovery",
        action="store_true",
        help="Required safety acknowledgement before restoring or cleaning a transaction",
    )
    return parser



# Promotion is not part of the public workflow. src/promotion.py implements the
# private repository's guarded results-promotion transaction and is not released,
# so these subcommands cannot run here. They are refused with an explanation
# rather than left to fail with ImportError.
_PUBLIC_RELEASE_DISABLED_COMMANDS = {
    "promote": (
        "Promotion replaces a live results/ tree from a validated run. There is "
        "no such tree in a public clone, and src/promotion.py is not part of "
        "this release."
    ),
    "recover-promotion": (
        "Promotion recovery inspects an interrupted promotion transaction, "
        "which cannot exist in a public clone."
    ),
}


def _refuse_if_disabled(command):
    message = _PUBLIC_RELEASE_DISABLED_COMMANDS.get(command)
    if message:
        raise SystemExit(
            f"'{command}' is not available in the public release. {message}\n"
            "Reproduce the analysis with: "
            "python scripts/run_production.py run-jama --execute-production"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _refuse_if_disabled(arguments.action)
    if arguments.action in {"plan", "plan-jama"}:
        placeholder = type(
            "PlanLayout",
            (),
            {
                "prepared_notebooks_dir": ROOT
                / "production_runs"
                / "<run_id>"
                / "prepared_notebooks",
                "executed_notebooks_dir": ROOT
                / "production_runs"
                / "<run_id>"
                / "executed_notebooks",
            },
        )()
        factory = (
            jama_command_specs
            if arguments.action == "plan-jama"
            else production_command_specs
        )
        for index, spec in enumerate(factory(ROOT, placeholder), 1):
            print(f"{index:02d}. {spec.command_id}")
        return 0

    if arguments.action in {"run", "run-jama"}:
        if not arguments.execute_production:
            raise ProductionError(
                "Production did not start: pass --execute-production explicitly"
            )
        git = require_clean_tracked_tree(ROOT)
        inputs = hash_inputs(ROOT)
        layout = create_run_layout(ROOT, str(git["commit"]))
        jama_scope = arguments.action == "run-jama"
        if jama_scope:
            if os.environ.get("MEDICAL_FAIRNESS_N_JOBS") != "8" or set(
                _THREAD_ENVIRONMENT.values()
            ) != {"8"}:
                raise ProductionError(
                    "JAMA production requires the fixed eight-thread entry-point profile"
                )
            os.environ["MEDICAL_FAIRNESS_PRODUCTION_SCOPE"] = "jama_manuscript"
        specs = (
            jama_command_specs(ROOT, layout)
            if jama_scope
            else production_command_specs(ROOT, layout)
        )
        manifest = execute_production_plan(
            layout,
            git,
            inputs,
            specs,
            include_retained_outputs=not jama_scope,
        )
        print(f"run_id={layout.run_id}")
        print(f"run_dir={layout.run_dir}")
        print(f"status={manifest['status']}")
        return 0 if manifest["status"] == "complete" else 1

    if arguments.action == "recover-promotion":
        recovery = recover_interrupted_promotion(
            ROOT, execute=arguments.execute_recovery
        )
        print(f"status={recovery['status']}")
        print(f"short_transaction_id={recovery['short_transaction_id']}")
        print(f"promotion_committed={str(recovery['promotion_committed']).lower()}")
        return 0

    run_dir = (
        arguments.run_dir
        if arguments.run_dir.is_absolute()
        else ROOT / arguments.run_dir
    ).resolve()
    plan_path = (
        arguments.disposition_plan
        if arguments.disposition_plan.is_absolute()
        else ROOT / arguments.disposition_plan
    ).resolve()
    if arguments.execute_promotion and arguments.verify_backup_filesystem:
        raise ProductionError(
            "Choose either backup verification or promotion execution, never both"
        )
    if arguments.verify_backup_filesystem:
        report = verify_promotion_backup_dry_run(
            ROOT,
            run_dir,
            plan_path,
            arguments.expected_promotion_code_commit,
        )
        print(f"status={report['status']}")
        print(f"source_run_id={report['source_run_id']}")
        print(f"backup_file_count={report['backup_file_count']}")
        print(f"backup_tree_sha256={report['backup_tree_sha256']}")
        print(
            "longest_destination_path_length="
            f"{report['longest_destination_path_length']}"
        )
        print(f"extended_length_paths_used={report['extended_length_paths_used']}")
        print("candidate_constructed=false")
        print("live_swap_performed=false")
        return 0
    if not arguments.execute_promotion:
        preview = build_promotion_dry_run(
            ROOT,
            run_dir,
            plan_path,
            arguments.expected_promotion_code_commit,
        )
        print(f"status={preview['status']}")
        print(f"source_run_id={preview['source_run_id']}")
        print(f"pre_promotion_file_count={preview['pre_promotion_file_count']}")
        print(f"staged_output_count={preview['staged_output_count']}")
        print(f"candidate_file_count={preview['candidate_file_count']}")
        for disposition, count in sorted(preview["disposition_counts"].items()):
            print(f"disposition_{disposition}={count}")
        print("promotion_performed=false")
        return 0
    result = promote_results(
        ROOT,
        run_dir,
        plan_path,
        arguments.expected_promotion_code_commit,
        execute=True,
    )
    print(f"status={result['status']}")
    print(f"promotion_receipt={result['receipt_path']}")
    print(f"promotion_backup={result['backup_path']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
