"""Staged production orchestration and provenance finalization.

This module deliberately imports only Python's standard library.  The command
entry point configures the deterministic thread environment before any child
process is allowed to import a numerical package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Callable, Iterable, Mapping, Sequence
import uuid

SCHEMA_VERSION = "2.2.0"
DEFAULT_SEED = 42
CONFIGURED_N_JOBS = int(os.environ.get("MEDICAL_FAIRNESS_N_JOBS", "1"))
if CONFIGURED_N_JOBS < 1:
    raise RuntimeError("MEDICAL_FAIRNESS_N_JOBS must be a positive integer")
THREAD_ENVIRONMENT = {
    "BLAS_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "OMP_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "MKL_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "OPENBLAS_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "NUMEXPR_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "BLIS_NUM_THREADS": str(CONFIGURED_N_JOBS),
    "VECLIB_MAXIMUM_THREADS": str(CONFIGURED_N_JOBS),
}
REQUIRED_MODEL_FAMILIES = (
    "logistic_regression",
    "l1_logistic_regression",
    "random_forest",
    "shallow_tree",
    "xgboost",
    "calibrated_ensemble",
)
PACKAGE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "matplotlib",
    "xgboost",
    "pyreadstat",
    "jupyter",
    "nbconvert",
)
INPUT_PATHS = (
    "data/cdc_diabetes.csv",
    "data/diabetes_130_hospitals.csv",
    "data/brfss2022_subset.csv",
    "data/nhanes_2017_2018.csv",
    "data/cchs_2019_2020_subset.csv",
)
RETAINED_OUTPUTS = (
    "extended/age_decomposition/NOTES.md",
    "extended/age_decomposition/age_stability_check_seed_results.csv",
    "extended/age_decomposition/age_stability_checks.csv",
    "extended/race_decomposition/NOTES.md",
)


class ProductionError(RuntimeError):
    """Controlled production protocol failure."""


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_id: str
    run_dir: Path
    staging_results: Path
    provenance_jsonl: Path
    run_manifest: Path
    run_manifest_sha256: Path
    logs_dir: Path
    prepared_notebooks_dir: Path
    executed_notebooks_dir: Path


@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    argv: tuple[str, ...]
    expected_outputs: tuple[str, ...] = ()
    expects_pipeline_provenance: bool = False
    notebook_source: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_deterministic_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Set every native thread limit before numerical libraries initialize."""
    target = os.environ if environment is None else environment
    for name, value in THREAD_ENVIRONMENT.items():
        target[name] = value
    target["PYTHONHASHSEED"] = "0"
    return {name: target[name] for name in (*THREAD_ENVIRONMENT, "PYTHONHASHSEED")}


def _strict_json_text(value: object, *, pretty: bool = False) -> str:
    kwargs = {"allow_nan": False, "sort_keys": True}
    if pretty:
        kwargs.update({"indent": 2})
    return json.dumps(value, **kwargs)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _promotion_filesystem_path(path: Path | str) -> str:
    """Use Windows extended-length syntax only at promotion read boundaries."""
    text = os.path.abspath(os.fspath(path))
    if os.name != "nt" or text.startswith("\\\\?\\") or len(text) < 240:
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _promotion_recursive_filesystem_path(path: Path | str) -> str:
    text = os.path.abspath(os.fspath(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _promotion_read_text(path: Path) -> str:
    with open(_promotion_filesystem_path(path), encoding="utf-8") as handle:
        return handle.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_promotion_filesystem_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise ProductionError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def require_clean_tracked_tree(root: Path) -> dict[str, object]:
    """Refuse production when any tracked path is staged or modified."""
    status_text = run_git(root, "status", "--porcelain=v1", "--untracked-files=no")
    status = [line for line in status_text.splitlines() if line]
    if status:
        raise ProductionError(
            "Refusing production from a dirty tracked tree: " + "; ".join(status)
        )
    commit = run_git(root, "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ProductionError(f"Git returned an invalid full commit hash: {commit!r}")
    return {
        "commit": commit,
        "branch": run_git(root, "branch", "--show-current"),
        "dirty": False,
        "tracked_status": status,
    }


def create_run_layout(
    root: Path,
    commit: str,
    *,
    run_id: str | None = None,
) -> RunLayout:
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{commit[:12]}_{uuid.uuid4().hex[:8]}"
    run_dir = root / "production_runs" / run_id
    if run_dir.exists():
        raise ProductionError(f"Production run directory already exists: {run_dir}")
    staging_results = run_dir / "staging" / "results"
    provenance_dir = run_dir / "provenance"
    logs_dir = run_dir / "logs"
    prepared = run_dir / "prepared_notebooks"
    executed = run_dir / "executed_notebooks"
    for directory in (staging_results, provenance_dir, logs_dir, prepared, executed):
        directory.mkdir(parents=True, exist_ok=False)
    return RunLayout(
        root=root,
        run_id=run_id,
        run_dir=run_dir,
        staging_results=staging_results,
        provenance_jsonl=provenance_dir / "provenance.jsonl",
        run_manifest=run_dir / "run_manifest.json",
        run_manifest_sha256=run_dir / "run_manifest.sha256",
        logs_dir=logs_dir,
        prepared_notebooks_dir=prepared,
        executed_notebooks_dir=executed,
    )


def hash_inputs(
    root: Path, paths: Iterable[str] = INPUT_PATHS
) -> list[dict[str, object]]:
    records = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise ProductionError(f"Required production input is missing: {relative}")
        records.append(
            {
                "path": Path(relative).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def build_command_environment(
    layout: RunLayout,
    git: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
    command_id: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    configure_deterministic_environment(environment)
    environment.update(
        {
            "MEDICAL_FAIRNESS_PROJECT_ROOT": str(layout.root.resolve()),
            "MEDICAL_FAIRNESS_RESULTS_DIR": str(layout.staging_results.resolve()),
            "MEDICAL_FAIRNESS_PROVENANCE_JSONL": str(layout.provenance_jsonl.resolve()),
            "MEDICAL_FAIRNESS_INPUT_HASHES_JSON": _strict_json_text(list(inputs)),
            "MEDICAL_FAIRNESS_COMMAND_ID": command_id,
            "MEDICAL_FAIRNESS_FAIL_FAST": "1",
            "MEDICAL_FAIRNESS_GIT_COMMIT": str(git["commit"]),
            "MEDICAL_FAIRNESS_GIT_BRANCH": str(git["branch"]),
            "MEDICAL_FAIRNESS_SOURCE_DIRTY": "1" if git["dirty"] else "0",
        }
    )
    return environment


def prepare_notebook_copy(source: Path, destination: Path) -> None:
    """Create a run-local notebook wired to the project and staged results."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    root_replacements = 0
    results_replacements = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        old_root = (
            "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) "
            'if (p / "src").is_dir())'
        )
        if old_root in text:
            text = text.replace(
                old_root,
                'ROOT = Path(__import__("os").environ["MEDICAL_FAIRNESS_PROJECT_ROOT"])',
            )
            root_replacements += 1
        if 'RESULTS_DIR = ROOT / "results"' in text:
            text = text.replace(
                'RESULTS_DIR = ROOT / "results"',
                'RESULTS_DIR = Path(__import__("os").environ['
                '"MEDICAL_FAIRNESS_RESULTS_DIR"])',
            )
            results_replacements += 1
        cell["source"] = text.splitlines(keepends=True)
    if root_replacements != 1 or results_replacements != 1:
        raise ProductionError(
            f"Notebook staging rewrite was not unique for {source}: "
            f"ROOT={root_replacements}, RESULTS_DIR={results_replacements}"
        )
    _atomic_write_text(destination, _strict_json_text(notebook, pretty=True) + "\n")


def _metadata_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _hash_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }


def detect_command_outputs(
    before: Mapping[str, tuple[int, int]],
    after: Mapping[str, tuple[int, int]],
) -> list[str]:
    return sorted(
        path for path, metadata in after.items() if before.get(path) != metadata
    )


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "binary"


def finalize_output_files(
    results_dir: Path,
    relative_paths: Iterable[str],
    command_id: str,
    *,
    disposition: str = "generated",
    finalized_at: str | None = None,
) -> list[dict[str, object]]:
    """Hash only outputs explicitly associated with the completed command."""
    finalized_at = finalized_at or utc_now()
    root = results_dir.resolve()
    records = []
    for relative in sorted(set(relative_paths)):
        path = (results_dir / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ProductionError(
                f"Output escapes staging results: {relative}"
            ) from exc
        if not path.is_file():
            raise ProductionError(f"Expected command output is missing: {relative}")
        records.append(
            {
                "path": f"results/{Path(relative).as_posix()}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "file_type": _file_type(path),
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "finalized_at_utc": finalized_at,
                "producing_command_id": command_id,
                "disposition": disposition,
            }
        )
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ProductionError(
                f"Invalid provenance JSONL at line {line_number}: {exc}"
            ) from exc
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    content = "".join(_strict_json_text(record) + "\n" for record in records)
    _atomic_write_text(path, content)


def append_provenance_record(path: Path, record: Mapping[str, object]) -> None:
    records = _read_jsonl(path)
    records.append(dict(record))
    _write_jsonl(path, records)


def finalize_pipeline_records(
    provenance_path: Path,
    command_id: str,
    outputs: Sequence[Mapping[str, object]],
    *,
    require_records: bool,
) -> int:
    records = _read_jsonl(provenance_path)
    finalized = 0
    for record in records:
        if (
            record.get("record_type") == "pipeline_analysis"
            and record.get("command_id") == command_id
        ):
            if not record.get("outputs_pending_command_finalization", False):
                raise ProductionError(
                    f"Pipeline record was already finalized for {command_id}"
                )
            record["outputs"] = [dict(output) for output in outputs]
            record["outputs_pending_command_finalization"] = False
            record["output_association_scope"] = "producing_command"
            record["finalized_at_utc"] = utc_now()
            finalized += 1
    if require_records and finalized == 0:
        raise ProductionError(
            f"Command {command_id} produced no pipeline-analysis provenance records"
        )
    if finalized:
        _write_jsonl(provenance_path, records)
    return finalized


def _append_command_record(
    provenance_path: Path,
    *,
    command_id: str,
    argv: Sequence[str],
    state: str,
    started_at: str,
    completed_at: str,
    return_code: int,
    outputs: Sequence[Mapping[str, object]],
    pipeline_record_count: int,
    error: str | None,
) -> None:
    append_provenance_record(
        provenance_path,
        {
            "record_type": "production_command",
            "command_id": command_id,
            "argv": list(argv),
            "state": state,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "return_code": return_code,
            "outputs": [dict(output) for output in outputs],
            "pipeline_record_count": pipeline_record_count,
            "error": error,
            "execution": {
                "n_jobs": CONFIGURED_N_JOBS,
                "thread_environment": dict(THREAD_ENVIRONMENT),
            },
        },
    )


def _manifest_payload_hash(manifest: Mapping[str, object]) -> str:
    payload = json.loads(_strict_json_text(manifest))
    integrity = payload.setdefault("integrity", {})
    integrity["manifest_payload_sha256"] = None
    return hashlib.sha256(_strict_json_text(payload).encode("utf-8")).hexdigest()


def checkpoint_manifest(layout: RunLayout, manifest: dict[str, object]) -> str:
    integrity = manifest.setdefault("integrity", {})
    integrity["final_manifest_sha256_location"] = layout.run_manifest_sha256.name
    integrity["self_hash_policy"] = (
        "manifest_payload_sha256 hashes canonical JSON with that field set to null; "
        "the detached run_manifest.sha256 hashes the final manifest bytes, avoiding "
        "an impossible embedded self-reference"
    )
    integrity["manifest_payload_sha256"] = _manifest_payload_hash(manifest)
    text = _strict_json_text(manifest, pretty=True) + "\n"
    _atomic_write_text(layout.run_manifest, text)
    final_hash = sha256_file(layout.run_manifest)
    _atomic_write_text(
        layout.run_manifest_sha256,
        f"{final_hash}  {layout.run_manifest.name}\n",
    )
    return final_hash


def initial_manifest(
    layout: RunLayout,
    git: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from .reproducibility_policy import manifest_reproducibility_policy

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": layout.run_id,
        "status": "incomplete",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "failed_analysis": None,
        "git": dict(git),
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "packages": package_versions(),
        },
        "inputs": [dict(item) for item in inputs],
        "execution": {
            "random_seed": DEFAULT_SEED,
            "n_jobs": CONFIGURED_N_JOBS,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "python_hash_seed": "0",
            "staging_results": str(layout.staging_results.relative_to(layout.root)),
            "production_scope": os.environ.get(
                "MEDICAL_FAIRNESS_PRODUCTION_SCOPE", "full_repository"
            ),
        },
        "protocol": {
            "model_selection": "five_fold_stratified_training_only_auroc",
            "candidate_families": list(REQUIRED_MODEL_FAMILIES),
            "family_eligibility": "finite mean CV AUROC > 0.5; every family required",
            "calibration": "five_fold_isotonic_fold_ensemble_ensemble_true",
            "threshold": "training_outcome_prevalence",
            "case_mix": "exploratory_point_estimates_no_confidence_intervals",
            "survey_weights": (
                "unweighted_sample_specific_analysis; no population-representative claims"
            ),
            "reproducibility_acceptance": manifest_reproducibility_policy(),
        },
        "commands": [],
        "validation": {
            "status": "not_run",
            "expected_outputs_complete": False,
            "provenance_finalized": False,
            "live_results_unchanged": None,
            "missing_outputs": [],
            "unassociated_outputs": [],
        },
        "outputs": [],
        "provenance": {
            "path": str(layout.provenance_jsonl.relative_to(layout.run_dir)),
            "sha256": None,
            "size_bytes": None,
            "finalized": False,
        },
        "integrity": {
            "manifest_payload_sha256": None,
            "final_manifest_sha256_location": layout.run_manifest_sha256.name,
            "self_hash_policy": "",
        },
        "promotion": {
            "eligible": False,
            "performed": False,
        },
    }


# Frozen per-comparison artifacts written by every primary analysis command:
# row-level held-out predictions, the discrimination/calibration table, the
# serialized fitted objects, and the artifact manifest.
FROZEN_ARTIFACT_SUFFIXES = (
    "_predictions.csv",
    "_performance.csv",
    "_model.joblib",
    "_manifest.json",
)


def _frozen_outputs(*slugs: str) -> tuple[str, ...]:
    return tuple(
        f"frozen/{slug}{suffix}"
        for slug in slugs
        for suffix in FROZEN_ARTIFACT_SUFFIXES
    )


def _primary_notebook_outputs(number: str, summary: str) -> tuple[str, ...]:
    stem = {
        "01": "cdc",
        "02": "diabetes130",
        "03": "brfss",
        "04": "nhanes",
        "05": "cchs",
    }[number]
    slug = {
        "01": "sex_cdc",
        "02": "sex_diabetes130",
        "03": "sex_brfss",
        "04": "sex_nhanes",
        "05": "sex_cchs",
    }[number]
    return (
        summary,
        f"{number}_{stem}_calibration.png",
        f"{number}_{stem}_case_mix_waterfall.png",
        f"{number}_{stem}_gap_npv.png",
        f"{number}_{stem}_gap_ppv.png",
        f"{number}_{stem}_gap_predicted_positive_rate.png",
        *_frozen_outputs(slug),
    )


# Dataset-name slugs used by the age and race runners' frozen exports.
_AGE_FROZEN_SLUGS = (
    "age_cdc_diabetes_brfss_2015",
    "age_diabetes_130_readmission",
    "age_brfss_2022_heart_disease",
    "age_nhanes_2017_18_diabetes",
    "age_cchs_2019_20_diabetes",
)
_RACE_FROZEN_SLUGS = (
    "race_diabetes_130_readmission",
    "race_brfss_2022_heart_disease",
    "race_nhanes_2017_18_diabetes",
    # Secondary BRFSS comparisons. They are reported in the Supplement, so
    # they are fitted analyses that must carry the same persisted evidence.
    "race_brfss_2022_heart_disease_hispanic_vs_white",
    "race_brfss_2022_heart_disease_multiracial_vs_white",
)


def production_command_specs(root: Path, layout: RunLayout) -> tuple[CommandSpec, ...]:
    python = sys.executable
    specs: list[CommandSpec] = [
        CommandSpec(
            "preflight_unit_tests",
            (python, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"),
        ),
        CommandSpec(
            "preflight_six_model_families",
            (
                python,
                "-B",
                "-c",
                "from src.pipeline import _build_models; "
                "models=_build_models(42); "
                "assert len(models)==6; print(','.join(models))",
            ),
        ),
        CommandSpec(
            "preflight_output_contracts",
            (python, "-B", "-m", "src.output_contracts"),
        ),
    ]
    notebook_definitions = (
        (
            "sex_cdc",
            "notebooks/01_cdc_diabetes.ipynb",
            _primary_notebook_outputs("01", "01_cdc_diabetes_summary.csv"),
        ),
        (
            "sex_diabetes130",
            "notebooks/02_diabetes130.ipynb",
            _primary_notebook_outputs("02", "02_diabetes130_summary.csv"),
        ),
        (
            "sex_brfss",
            "notebooks/03_brfss.ipynb",
            _primary_notebook_outputs("03", "03_brfss_summary.csv"),
        ),
        (
            "sex_nhanes",
            "notebooks/04_nhanes.ipynb",
            _primary_notebook_outputs("04", "04_nhanes_summary.csv"),
        ),
        (
            "sex_cchs",
            "notebooks/05_cchs.ipynb",
            _primary_notebook_outputs("05", "05_cchs_summary.csv"),
        ),
        (
            "sex_pooled",
            "notebooks/06_pooled_analysis.ipynb",
            (
                "06_pooled_attenuation_table.csv",
                "06_pooled_sensitivity_gap.png",
                "06_pooled_ppv_gap_raw.png",
                "06_pooled_ppv_gap_adjusted.png",
                "06_pooled_npv_gap_raw.png",
                "06_pooled_npv_gap_adjusted.png",
                "06_pooled_ppr_gap_raw.png",
                "06_pooled_ppr_gap_adjusted.png",
            ),
        ),
    )
    for command_id, notebook_source, expected in notebook_definitions:
        specs.append(
            CommandSpec(
                command_id,
                (),
                tuple(expected),
                expects_pipeline_provenance=command_id != "sex_pooled",
                notebook_source=notebook_source,
            )
        )
    specs.extend(
        [
            CommandSpec(
                "age_decomposition",
                (python, "-B", "run_age_decomposition.py"),
                tuple(
                    f"extended/age_decomposition/{name}"
                    for name in (
                        "age_raw_gaps.csv",
                        "age_bootstrap.csv",
                        "age_shapley.csv",
                        "age_casemix_waterfall.csv",
                        "age_ipw_adjustment.csv",
                        "age_all_bands.csv",
                        "age_meta_analysis_sensitivity_gap.csv",
                        "age_meta_analysis_per_study.csv",
                    )
                ),
                True,
            ),
            CommandSpec(
                "race_decomposition",
                (python, "-B", "run_race_decomposition.py"),
                tuple(
                    f"extended/race_decomposition/{name}"
                    for name in (
                        "race_raw_gaps.csv",
                        "race_bootstrap.csv",
                        "race_shapley.csv",
                        "race_casemix_waterfall.csv",
                        "race_ipw_adjustment.csv",
                        "race_all_groups.csv",
                        "race_secondary_comparisons.csv",
                        "race_stability_checks.csv",
                        "race_meta_analysis_sensitivity_gap.csv",
                        "race_meta_analysis_per_study.csv",
                    )
                ),
                True,
            ),
        ]
    )

    def extended(
        command_id: str, filename: str, outputs: Sequence[str], *, pipeline=True
    ):
        specs.append(
            CommandSpec(
                command_id,
                (python, "-B", f"extended_robustness_scripts/{filename}"),
                tuple(f"extended/{name}" for name in outputs),
                pipeline,
            )
        )

    extended(
        "extended_core",
        "run_extended_analyses.py",
        (
            "fn_regression_waterfall.csv",
            "fp_regression_waterfall.csv",
            "confusion_matrices.csv",
            "score_dist_positives.csv",
            "score_dist_negatives.csv",
            "repeated_splits.csv",
            "threshold_sweep.csv",
            "shapley_decomposition.csv",
            "empirical_prevalence_matching.csv",
            "synthetic_controls.csv",
            "figures/repeated_splits_sens_gap.png",
            *(
                f"figures/threshold_sweep_{key}.png"
                for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
            ),
            *(
                f"figures/shapley_{key}.png"
                for key in ("prevalence", "sensitivity", "specificity")
            ),
        ),
    )
    extended(
        "extended_2_3",
        "run_extended_2_3.py",
        (
            "covariate_balance_table.csv",
            "covariate_missingness_raw.csv",
            "data_quality_table.csv",
            "survey_weighted_vs_unweighted.csv",
            "diabetes130_deduplication_comparison.csv",
            "diabetes130_cluster_robust_ci.csv",
        ),
    )
    extended(
        "extended_11_12_27_28",
        "run_extended_11_12_27_28.py",
        (
            "risk_decile_error_analysis.csv",
            "prevalence_sweep_curves.csv",
            "covariate_standardized_error_gaps.csv",
            "synthetic_prevalence_only_control.csv",
            "synthetic_prevalence_only_control_summary.csv",
            *(
                f"figures/risk_decile_{key}.png"
                for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
            ),
            *(
                f"figures/prevalence_sweep_{key}.png"
                for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
            ),
            "figures/synthetic_prevalence_only_control.png",
        ),
    )
    extended(
        "extended_13_20_34",
        "run_extended_13_20_34.py",
        (
            "calibration_curves_by_subgroup.csv",
            "recalibration_comparison_subgroup.csv",
            "calibration_method_robustness.csv",
            "calibration_method_robustness_summary.csv",
            *(
                f"figures/calibration_curve_ci_{key}.png"
                for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
            ),
        ),
    )
    extended(
        "extended_16_19",
        "run_extended_16_19.py",
        (
            "equal_specificity_threshold.csv",
            "equal_ppv_threshold.csv",
            "equal_demographic_parity_threshold.csv",
            "pareto_frontier.csv",
            *(
                f"figures/pareto_frontier_{key}.png"
                for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
            ),
        ),
    )
    extended(
        "extended_29_37_38",
        "run_extended_29_37_38.py",
        (
            "subgroup_covariate_interactions.csv",
            "balanced_training_analysis.csv",
            "additional_protected_attributes.csv",
            "intersectional_stratified_sex_gap.csv",
        ),
    )
    extended(
        "extended_31_33",
        "run_extended_31_33.py",
        ("nested_cv_results.csv", "hyperparameter_robustness.csv"),
        # This runner intentionally fits its own nested-CV and
        # hyperparameter models. It receives command-level provenance and
        # output hashes, but does not call the shared fairness pipeline and
        # therefore must not be required to fabricate pipeline records.
        pipeline=False,
    )
    extended(
        "extended_32_35_36",
        "run_extended_32_35_36.py",
        (
            "model_family_robustness.csv",
            "threshold_policy_robustness.csv",
            "protected_attribute_inclusion.csv",
        ),
    )
    extended(
        "extended_39_40_41",
        "run_extended_39_40_41.py",
        (
            "temporal_validation_data_availability.csv",
            "temporal_validation_brfss_cross_wave.csv",
            "diabetes130_site_level_validation.csv",
            "outcome_family_replication_synthesis.csv",
        ),
        pipeline=False,
    )
    extended(
        "extended_48_49_50",
        "run_extended_48_49_50.py",
        (
            "protected_label_permutation_summary.csv",
            "protected_label_permutation_null.csv",
            "outcome_label_permutation_summary.csv",
            "outcome_label_permutation_null.csv",
            "null_feature_model_comparison.csv",
        ),
    )
    extended(
        "extended_42_44_45",
        "run_extended_42_44_45.py",
        (
            "meta_analysis_sensitivity_gap_per_study.csv",
            "meta_analysis_sensitivity_gap_leave_one_out.csv",
            "meta_analysis_sensitivity_gap_summary.csv",
            "bh_fdr_correction_all_tests.csv",
            "bh_fdr_correction_primary_endpoints.csv",
            "equivalence_tests.csv",
        ),
        pipeline=False,
    )
    specs.append(
        CommandSpec(
            "postflight_unit_tests",
            (python, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"),
        )
    )
    return tuple(specs)


def jama_command_specs(root: Path, layout: RunLayout) -> tuple[CommandSpec, ...]:
    """Fixed, manuscript-scoped JAMA production order.

    This deliberately excludes repository-wide robustness commands that are
    not reported in the current main manuscript or Supplement. The age and
    race runners use their ``--jama-only`` save mode so obsolete meta-analysis
    and optional auxiliary files are not regenerated into this staging tree.
    """
    python = sys.executable
    specs: list[CommandSpec] = [
        CommandSpec(
            "preflight_unit_tests",
            (python, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"),
        ),
        CommandSpec(
            "preflight_six_model_families",
            (
                python,
                "-B",
                "-c",
                "from src.pipeline import _build_models; "
                "models=_build_models(42); assert len(models)==6; print(','.join(models))",
            ),
        ),
        CommandSpec(
            "preflight_output_contracts",
            (python, "-B", "-m", "src.output_contracts"),
        ),
        CommandSpec(
            "preflight_eight_thread_reproducibility",
            (python, "-B", "-m", "src.reproducibility_preflight"),
        ),
    ]
    notebook_definitions = (
        (
            "sex_cdc",
            "notebooks/01_cdc_diabetes.ipynb",
            _primary_notebook_outputs("01", "01_cdc_diabetes_summary.csv"),
        ),
        (
            "sex_diabetes130",
            "notebooks/02_diabetes130.ipynb",
            _primary_notebook_outputs("02", "02_diabetes130_summary.csv"),
        ),
        (
            "sex_brfss",
            "notebooks/03_brfss.ipynb",
            _primary_notebook_outputs("03", "03_brfss_summary.csv"),
        ),
        (
            "sex_nhanes",
            "notebooks/04_nhanes.ipynb",
            _primary_notebook_outputs("04", "04_nhanes_summary.csv"),
        ),
        (
            "sex_cchs",
            "notebooks/05_cchs.ipynb",
            _primary_notebook_outputs("05", "05_cchs_summary.csv"),
        ),
    )
    for command_id, notebook_source, expected in notebook_definitions:
        specs.append(
            CommandSpec(command_id, (), tuple(expected), True, notebook_source)
        )
    specs.extend(
        [
            CommandSpec(
                "age_decomposition",
                (python, "-B", "run_age_decomposition.py", "--jama-only"),
                tuple(
                    f"extended/age_decomposition/{name}"
                    for name in (
                        "age_raw_gaps.csv",
                        "age_bootstrap.csv",
                        "age_casemix_waterfall.csv",
                        "age_ipw_adjustment.csv",
                    )
                )
                + _frozen_outputs(*_AGE_FROZEN_SLUGS),
                True,
            ),
            CommandSpec(
                "race_decomposition",
                (python, "-B", "run_race_decomposition.py", "--jama-only"),
                tuple(
                    f"extended/race_decomposition/{name}"
                    for name in (
                        "race_raw_gaps.csv",
                        "race_bootstrap.csv",
                        "race_casemix_waterfall.csv",
                        "race_ipw_adjustment.csv",
                        "race_secondary_comparisons.csv",
                        "race_stability_checks.csv",
                    )
                )
                + _frozen_outputs(*_RACE_FROZEN_SLUGS),
                True,
            ),
            CommandSpec(
                "jama_supplement_analyses",
                (python, "-B", "scripts/run_jama_supplement.py"),
                tuple(
                    f"extended/jama_supplement/{name}"
                    for name in (
                        "threshold_sweep.csv",
                        "threshold_policy.csv",
                        "prevalence_standardized_metrics.csv",
                        "case_mix_point_estimates.csv",
                        *(
                            f"figures/threshold_sweep_{key}.png"
                            for key in ("cdc", "d130", "brfss", "nhanes", "cchs")
                        ),
                    )
                ),
                True,
            ),
            CommandSpec(
                "jama_reporting_sources",
                (python, "-B", "scripts/build_jama_reporting_sources.py"),
                tuple(
                    f"jama_reporting/{name}"
                    for name in (
                        "table1_source.csv",
                        "etable1_subgroup_metrics_source.csv",
                        "missing_data_flow.csv",
                        "figure1_source_values.csv",
                        "etable_model_performance_source.csv",
                        "etable_calibration_source.csv",
                    )
                ),
                False,
            ),
            CommandSpec(
                "postflight_unit_tests",
                (python, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"),
            ),
        ]
    )
    return tuple(specs)


def _materialize_notebook_command(spec: CommandSpec, layout: RunLayout) -> CommandSpec:
    if not spec.notebook_source:
        return spec
    source = layout.root / spec.notebook_source
    prepared = layout.prepared_notebooks_dir / source.name
    executed_name = f"{source.stem}.executed.ipynb"
    prepare_notebook_copy(source, prepared)
    return CommandSpec(
        spec.command_id,
        (
            sys.executable,
            "-B",
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(prepared),
            "--output",
            executed_name,
            "--output-dir",
            str(layout.executed_notebooks_dir),
            "--ExecutePreprocessor.timeout=-1",
        ),
        spec.expected_outputs,
        spec.expects_pipeline_provenance,
        spec.notebook_source,
    )


CommandExecutor = Callable[[CommandSpec, Mapping[str, str], Path, Path, Path], int]


def subprocess_executor(
    spec: CommandSpec,
    environment: Mapping[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(
            list(spec.argv),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    return completed.returncode


def _copy_retained_outputs(layout: RunLayout) -> list[str]:
    copied = []
    live_results = layout.root / "results"
    for relative in RETAINED_OUTPUTS:
        source = live_results / relative
        destination = layout.staging_results / relative
        if not source.is_file():
            raise ProductionError(f"Required retained evidence is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative)
    return copied


def _finalize_provenance_summary(
    layout: RunLayout, manifest: dict[str, object]
) -> None:
    provenance = manifest["provenance"]
    if layout.provenance_jsonl.exists():
        provenance["sha256"] = sha256_file(layout.provenance_jsonl)
        provenance["size_bytes"] = layout.provenance_jsonl.stat().st_size
    else:
        provenance["sha256"] = None
        provenance["size_bytes"] = 0


def execute_production_plan(
    layout: RunLayout,
    git: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
    specs: Sequence[CommandSpec],
    *,
    executor: CommandExecutor = subprocess_executor,
    include_retained_outputs: bool = True,
) -> dict[str, object]:
    """Execute a staged plan; never promote or write to the live results tree."""
    manifest = initial_manifest(layout, git, inputs)
    checkpoint_manifest(layout, manifest)
    live_results = layout.root / "results"
    live_before = _hash_snapshot(live_results)
    output_registry: dict[str, dict[str, object]] = {}
    expected_all: set[str] = set()

    try:
        if include_retained_outputs:
            command_id = "retain_verified_evidence"
            started = utc_now()
            retained = _copy_retained_outputs(layout)
            records = finalize_output_files(
                layout.staging_results,
                retained,
                command_id,
                disposition="retained",
            )
            for record in records:
                output_registry[str(record["path"])] = record
            completed = utc_now()
            _append_command_record(
                layout.provenance_jsonl,
                command_id=command_id,
                argv=("internal:copy_retained_outputs",),
                state="succeeded",
                started_at=started,
                completed_at=completed,
                return_code=0,
                outputs=records,
                pipeline_record_count=0,
                error=None,
            )
            manifest["commands"].append(
                {
                    "command_id": command_id,
                    "argv": ["internal:copy_retained_outputs"],
                    "state": "succeeded",
                    "started_at_utc": started,
                    "completed_at_utc": completed,
                    "return_code": 0,
                    "outputs": records,
                    "error": None,
                }
            )
            expected_all.update(retained)
            checkpoint_manifest(layout, manifest)

        for original_spec in specs:
            spec = _materialize_notebook_command(original_spec, layout)
            expected_all.update(spec.expected_outputs)
            started = utc_now()
            attempt = {
                "command_id": spec.command_id,
                "argv": list(spec.argv),
                "state": "running",
                "started_at_utc": started,
                "completed_at_utc": None,
                "return_code": None,
                "outputs": [],
                "error": None,
            }
            manifest["commands"].append(attempt)
            checkpoint_manifest(layout, manifest)
            before = _metadata_snapshot(layout.staging_results)
            environment = build_command_environment(
                layout, git, inputs, spec.command_id
            )
            stdout_path = layout.logs_dir / f"{spec.command_id}.stdout.log"
            stderr_path = layout.logs_dir / f"{spec.command_id}.stderr.log"
            execution_error = None
            try:
                return_code = executor(
                    spec, environment, layout.root, stdout_path, stderr_path
                )
            except BaseException as exc:
                return_code = -1
                execution_error = f"{type(exc).__name__}: {exc}"
            after = _metadata_snapshot(layout.staging_results)
            changed = detect_command_outputs(before, after)
            output_records = finalize_output_files(
                layout.staging_results, changed, spec.command_id
            )
            for record in output_records:
                output_registry[str(record["path"])] = record
            missing = [
                relative
                for relative in spec.expected_outputs
                if not (layout.staging_results / relative).is_file()
            ]
            error = execution_error
            if return_code and error is None:
                error = f"Command exited with status {return_code}"
            if error is None and missing:
                error = "Missing expected outputs: " + ", ".join(missing)
            pipeline_count = finalize_pipeline_records(
                layout.provenance_jsonl,
                spec.command_id,
                output_records,
                require_records=False,
            )
            if (
                error is None
                and spec.expects_pipeline_provenance
                and pipeline_count == 0
            ):
                error = "Required pipeline-analysis provenance record was not created"
            if error is None and _hash_snapshot(live_results) != live_before:
                error = "Live results tree changed during staged production"
            completed = utc_now()
            state = "succeeded" if error is None else "failed"
            _append_command_record(
                layout.provenance_jsonl,
                command_id=spec.command_id,
                argv=spec.argv,
                state=state,
                started_at=started,
                completed_at=completed,
                return_code=return_code,
                outputs=output_records,
                pipeline_record_count=pipeline_count,
                error=error,
            )
            attempt.update(
                {
                    "state": state,
                    "completed_at_utc": completed,
                    "return_code": return_code,
                    "outputs": output_records,
                    "error": error,
                    "stdout_log": str(stdout_path.relative_to(layout.run_dir)),
                    "stderr_log": str(stderr_path.relative_to(layout.run_dir)),
                }
            )
            _finalize_provenance_summary(layout, manifest)
            if error is not None:
                manifest["status"] = "failed"
                manifest["failed_analysis"] = spec.command_id
                manifest["completed_at_utc"] = completed
                manifest["outputs"] = list(output_registry.values())
                manifest["validation"]["status"] = "failed"
                manifest["validation"]["missing_outputs"] = missing
                manifest["validation"]["live_results_unchanged"] = (
                    _hash_snapshot(live_results) == live_before
                )
                checkpoint_manifest(layout, manifest)
                return manifest
            checkpoint_manifest(layout, manifest)

        actual = _hash_snapshot(layout.staging_results)
        missing_all = sorted(
            relative for relative in expected_all if relative not in actual
        )
        unassociated = sorted(
            relative
            for relative in actual
            if f"results/{relative}" not in output_registry
        )
        pending = [
            record
            for record in _read_jsonl(layout.provenance_jsonl)
            if record.get("outputs_pending_command_finalization")
        ]
        live_unchanged = _hash_snapshot(live_results) == live_before
        hash_mismatches = []
        for relative, (size, digest) in actual.items():
            record = output_registry.get(f"results/{relative}")
            if record and (record["size_bytes"] != size or record["sha256"] != digest):
                hash_mismatches.append(relative)
        passed = (
            not (missing_all or unassociated or pending or hash_mismatches)
            and live_unchanged
        )
        manifest["validation"] = {
            "status": "passed" if passed else "failed",
            "expected_outputs_complete": not missing_all,
            "provenance_finalized": not pending,
            "live_results_unchanged": live_unchanged,
            "missing_outputs": missing_all,
            "unassociated_outputs": unassociated,
            "hash_mismatches": hash_mismatches,
        }
        manifest["outputs"] = [
            output_registry[path] for path in sorted(output_registry)
        ]
        manifest["completed_at_utc"] = utc_now()
        manifest["status"] = "complete" if passed else "failed"
        if not passed:
            manifest["failed_analysis"] = "post_run_integrity_validation"
        manifest["provenance"]["finalized"] = not pending
        _finalize_provenance_summary(layout, manifest)
        manifest["promotion"]["eligible"] = passed
        checkpoint_manifest(layout, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = utc_now()
        if manifest["failed_analysis"] is None:
            running = [
                item for item in manifest["commands"] if item["state"] == "running"
            ]
            manifest["failed_analysis"] = (
                running[-1]["command_id"] if running else "production_orchestrator"
            )
        for item in manifest["commands"]:
            if item["state"] == "running":
                item["state"] = "failed"
                item["completed_at_utc"] = manifest["completed_at_utc"]
                item["error"] = f"{type(exc).__name__}: {exc}"
        manifest["validation"]["status"] = "failed"
        manifest["validation"]["live_results_unchanged"] = (
            _hash_snapshot(live_results) == live_before
        )
        manifest["outputs"] = list(output_registry.values())
        _finalize_provenance_summary(layout, manifest)
        checkpoint_manifest(layout, manifest)
        return manifest


def _promotion_hash_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """Long-path-safe immutable-tree snapshot used only by promotion preflight."""
    filesystem_root = _promotion_recursive_filesystem_path(root)
    if not os.path.isdir(filesystem_root):
        return {}
    records: dict[str, tuple[int, str]] = {}
    for directory, _directory_names, file_names in os.walk(
        filesystem_root, followlinks=False
    ):
        relative_directory = os.path.relpath(directory, filesystem_root)
        for name in file_names:
            relative = (
                name
                if relative_directory == "."
                else Path(relative_directory, name).as_posix()
            )
            path = root / Path(relative)
            info = os.stat(_promotion_filesystem_path(path))
            records[relative] = (info.st_size, sha256_file(path))
    return records


def validate_promotion_eligibility(run_dir: Path) -> dict[str, object]:
    """Validate immutable run evidence before disposition planning.

    Promotion has stricter evidence requirements than ordinary run inspection:
    both manifest hashes, the strict provenance stream, and every staged output
    association must agree before the disposition layer is allowed to proceed.
    """
    run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    manifest_path = run_dir / "run_manifest.json"
    detached_path = run_dir / "run_manifest.sha256"
    if not os.path.isfile(
        _promotion_filesystem_path(manifest_path)
    ) or not os.path.isfile(_promotion_filesystem_path(detached_path)):
        raise ProductionError(
            f"Run manifest or detached hash is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(
            _promotion_read_text(manifest_path),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProductionError(
                    f"Strict run manifest rejected nonfinite value: {value}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionError(f"Run manifest is not strict JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProductionError("Run manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProductionError(
            f"Refusing promotion: manifest schema is not {SCHEMA_VERSION}"
        )
    detached_parts = _promotion_read_text(detached_path).split()
    if (
        len(detached_parts) != 2
        or detached_parts[1] != manifest_path.name
        or detached_parts[0] != sha256_file(manifest_path)
    ):
        raise ProductionError(
            "Refusing promotion: run-manifest detached hash is invalid"
        )
    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("manifest_payload_sha256") != _manifest_payload_hash(manifest)
        or integrity.get("final_manifest_sha256_location") != detached_path.name
    ):
        raise ProductionError(
            "Refusing promotion: embedded manifest integrity is invalid"
        )
    if manifest.get("status") != "complete":
        raise ProductionError("Refusing promotion: run status is not complete")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ProductionError("Refusing promotion: validation summary is invalid")
    if validation.get("status") != "passed":
        raise ProductionError("Refusing promotion: validation did not pass")
    if not validation.get("expected_outputs_complete"):
        raise ProductionError("Refusing promotion: expected outputs are incomplete")
    if not validation.get("provenance_finalized"):
        raise ProductionError("Refusing promotion: provenance is not finalized")
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict) or not promotion.get("eligible"):
        raise ProductionError("Refusing promotion: manifest is not promotion-eligible")
    if promotion.get("performed"):
        raise ProductionError(
            "Refusing promotion: source manifest already records promotion"
        )

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("finalized") is not True:
        raise ProductionError("Refusing promotion: provenance summary is not finalized")
    raw_provenance_path = provenance.get("path")
    if not isinstance(raw_provenance_path, str) or not raw_provenance_path:
        raise ProductionError("Refusing promotion: provenance path is invalid")
    normalized_provenance_path = raw_provenance_path.replace("\\", "/")
    raw_path = PurePosixPath(normalized_provenance_path)
    if (
        ":" in raw_provenance_path
        or raw_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_path.parts)
    ):
        raise ProductionError("Refusing promotion: provenance escaped the run")
    provenance_path = run_dir / Path(*raw_path.parts)
    if not os.path.isfile(_promotion_filesystem_path(provenance_path)):
        raise ProductionError("Refusing promotion: provenance stream is missing")
    try:
        provenance_size = int(provenance.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ProductionError("Refusing promotion: provenance size is invalid") from exc
    if (
        provenance.get("sha256") != sha256_file(provenance_path)
        or provenance_size
        != os.stat(_promotion_filesystem_path(provenance_path)).st_size
    ):
        raise ProductionError("Refusing promotion: provenance hash or size is invalid")
    provenance_records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        _promotion_read_text(provenance_path).splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ProductionError(
                        "Strict provenance rejected nonfinite value at line "
                        f"{line_number}: {value}"
                    )
                ),
            )
        except json.JSONDecodeError as exc:
            raise ProductionError(
                f"Refusing promotion: invalid provenance line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ProductionError(
                f"Refusing promotion: provenance line {line_number} is not an object"
            )
        provenance_records.append(record)
    if any(
        record.get("outputs_pending_command_finalization") is True
        for record in provenance_records
    ):
        raise ProductionError("Refusing promotion: provenance contains pending records")

    # Normal layout is fixed; do not trust a manifest path for filesystem mutation.
    staging = run_dir / "staging" / "results"
    actual = _promotion_hash_snapshot(staging)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ProductionError("Refusing promotion: manifest outputs are invalid")
    expected: dict[str, tuple[int, str]] = {}
    folded_paths: set[str] = set()
    for item in outputs:
        if not isinstance(item, dict):
            raise ProductionError("Refusing promotion: output record is not an object")
        full_path = str(item.get("path", ""))
        if not full_path.startswith("results/"):
            raise ProductionError(
                f"Refusing promotion: unsafe output path {full_path!r}"
            )
        relative = full_path.removeprefix("results/")
        pure = Path(relative)
        if (
            not relative
            or "\\" in relative
            or ":" in relative
            or "\x00" in relative
            or pure.is_absolute()
            or ".." in pure.parts
            or any(part.rstrip(" .") != part for part in pure.parts)
            or relative.casefold() in folded_paths
        ):
            raise ProductionError(
                f"Refusing promotion: duplicate or unsafe output {relative!r}"
            )
        folded_paths.add(relative.casefold())
        digest = str(item.get("sha256", ""))
        producer = str(item.get("producing_command_id", ""))
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            or not producer
        ):
            raise ProductionError(
                f"Refusing promotion: incomplete output association for {relative}"
            )
        try:
            size = int(item.get("size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ProductionError(
                f"Refusing promotion: invalid output size for {relative}"
            ) from exc
        expected[relative] = (size, digest)
    if actual != expected:
        raise ProductionError("Refusing promotion: staged output hashes do not match")
    return manifest


def build_promotion_dry_run(
    root: Path,
    run_dir: Path,
    plan_path: Path,
    expected_promotion_code_commit: str,
    *,
    git_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and report a disposition-driven promotion without writing."""
    from .promotion import build_promotion_dry_run as implementation

    return implementation(
        root,
        run_dir,
        plan_path,
        expected_promotion_code_commit,
        git_state=git_state,
    )


def verify_promotion_backup_dry_run(
    root: Path,
    run_dir: Path,
    plan_path: Path,
    expected_promotion_code_commit: str,
    *,
    git_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create and verify an ephemeral full backup without a candidate or swap."""
    from .promotion import verify_promotion_backup_dry_run as implementation

    return implementation(
        root,
        run_dir,
        plan_path,
        expected_promotion_code_commit,
        git_state=git_state,
    )


def recover_interrupted_promotion(
    root: Path,
    *,
    execute: bool = False,
) -> dict[str, object]:
    """Inspect or explicitly recover an interrupted guarded promotion."""
    from .promotion import recover_interrupted_promotion as implementation

    return implementation(root, execute=execute)


def promote_results(
    root: Path,
    run_dir: Path,
    plan_path: Path,
    expected_promotion_code_commit: str,
    *,
    execute: bool = False,
    git_state: Mapping[str, object] | None = None,
    post_swap_validator: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Dry-run or execute the hardened disposition-driven promotion."""
    from .promotion import promote_results as implementation

    return implementation(
        root,
        run_dir,
        plan_path,
        expected_promotion_code_commit,
        execute=execute,
        git_state=git_state,
        post_swap_validator=post_swap_validator,
    )
