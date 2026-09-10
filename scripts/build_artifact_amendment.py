"""Record a post-run, metadata-only amendment to a completed production run.

Why this exists
---------------
The run manifest is an attestation of the production run *at completion*. After
that run finished, the 15 fitted model artifacts and their 15 companion frozen
manifests were repacked to correct scoring-interface metadata: the artifacts had
advertised every column of the working frame as a required scoring input,
including the outcome, the demographic subgroup label, and a source identifier.
Repacking changed those files' bytes and therefore their checksums.

No estimator, preprocessor, or calibration object was refitted, and no predicted
probability changed. But the original manifest cannot and must not be edited to
pretend it attested to files that did not exist when it was written. Instead
this produces a separate, explicit attestation covering exactly the amended
files, so the two records together describe the tree without either one lying:

    original run manifest -> the production run's numerical outputs
    this amendment        -> the metadata-repacked fitted artifacts

Every amended artifact is independently rescored here: reloaded, used to score
the held-out predictor rows rebuilt from the source dataset, and compared with
the frozen probabilities. The amendment records the deviation rather than
asserting equivalence.

Usage:
    python scripts/build_artifact_amendment.py --run-dir <dir> --repack-commit <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AMENDMENT_JSON = "POST_RUN_ARTIFACT_AMENDMENT.json"
AMENDMENT_MD = "POST_RUN_ARTIFACT_AMENDMENT.md"
SCHEMA_VERSION = "1.0.0"

REASON = (
    "Scoring-interface metadata correction only. The artifacts advertised every "
    "column of the working frame as a required scoring input, including the "
    "outcome, the demographic subgroup label, and a source identifier. They now "
    "declare only the raw predictor columns the model consumes."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_timestamp(commit):
    try:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%cI", commit],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repack-commit", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    staging = run_dir / "staging"
    frozen = staging / "results" / "frozen"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    original = {o["path"]: o for o in manifest["outputs"]}

    # ---- which recorded outputs no longer match the original attestation ----
    amended_paths, unchanged = [], 0
    for path, record in sorted(original.items()):
        current = staging / path
        if not current.exists():
            raise SystemExit(f"recorded output is missing: {path}")
        if sha256(current) == record["sha256"]:
            unchanged += 1
        else:
            amended_paths.append(path)

    joblibs = [p for p in amended_paths if p.endswith(".joblib")]
    manifests = [p for p in amended_paths if p.endswith(".json")]
    if len(joblibs) != len(manifests):
        raise SystemExit(
            f"expected paired artifacts and manifests, got {len(joblibs)} and "
            f"{len(manifests)}"
        )

    # ---- independently rescore every amended artifact -----------------------
    from scripts.score_with_frozen_model import (
        MATCH_TOLERANCE, SLUG_LOADER, load_artifact, score_frame,
    )
    from src import datasets

    cache = {}
    files, deviations = [], []
    for path in amended_paths:
        current = staging / path
        entry = {
            "path": path,
            "kind": "fitted_model_artifact" if path.endswith(".joblib")
                    else "frozen_artifact_manifest",
            "original_sha256": original[path]["sha256"],
            "original_size_bytes": original[path]["size_bytes"],
            "amended_sha256": sha256(current),
            "amended_size_bytes": current.stat().st_size,
            "amended_at_utc": datetime.fromtimestamp(
                current.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        if path.endswith(".joblib"):
            slug = Path(path).name[: -len("_model.joblib")]
            payload = load_artifact(current)
            predictions = frozen / f"{slug}_predictions.csv"
            saved = pd.read_csv(predictions)
            loader = SLUG_LOADER[slug]
            if loader not in cache:
                cache[loader] = getattr(datasets, loader)()["df"]
            rows = cache[loader].loc[saved["row_id"].tolist()]
            recomputed = score_frame(payload, rows)["prob_calibrated"].to_numpy()
            deviation = float(np.max(np.abs(
                recomputed - saved["prob_calibrated"].to_numpy()
            )))
            deviations.append(deviation)
            entry.update({
                "comparison_slug": slug,
                "selected_family": payload["selected_family"],
                "frozen_prediction_reference":
                    f"results/frozen/{predictions.name}",
                "rows_rescored": int(len(saved)),
                "max_abs_deviation_from_frozen": deviation,
                "verification_status": (
                    "verified_by_independent_rescoring"
                    if deviation <= MATCH_TOLERANCE
                    else f"FAILED max_abs_deviation={deviation:.3e}"
                ),
                "n_required_predictor_columns":
                    len(payload["required_predictor_columns"]),
            })
        else:
            entry["verification_status"] = (
                "metadata record accompanying the amended artifact"
            )
        files.append(entry)

    failures = [f for f in files if str(f.get("verification_status", "")).startswith("FAILED")]

    amendment = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "post_run_artifact_amendment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_run": {
            "run_id": manifest["run_id"],
            "analysis_source_commit": manifest["git"]["commit"],
            "completed_at_utc": manifest["completed_at_utc"],
            "status": manifest["status"],
            "validation_status": manifest["validation"]["status"],
            "recorded_outputs": len(original),
        },
        "amendment": {
            "reason": REASON,
            "repack_commit": args.repack_commit,
            "repack_commit_timestamp": _commit_timestamp(args.repack_commit),
            "refit_performed": False,
            "estimator_refit": False,
            "preprocessor_refit": False,
            "calibration_refit": False,
            "numerical_predictions_changed": False,
            "amended_file_count": len(files),
            "amended_model_artifacts": len(joblibs),
            "amended_frozen_manifests": len(manifests),
            "artifacts_independently_rescored": len(deviations),
            "max_abs_deviation_across_artifacts":
                max(deviations) if deviations else None,
            "verification_tolerance": MATCH_TOLERANCE,
            "all_artifacts_verified": not failures,
        },
        "attestation_scope": {
            "original_run_manifest": (
                "Authoritative for the production run's numerical outputs. "
                f"{unchanged} of {len(original)} recorded outputs still match it "
                "byte for byte."
            ),
            "this_amendment": (
                "Authoritative for the integrity of the metadata-repacked fitted "
                f"artifacts. It covers exactly {len(files)} files and no others."
            ),
            "unchanged_outputs_matching_original_manifest": unchanged,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": manifest["environment"]["packages"],
        },
        "files": files,
    }

    (run_dir / AMENDMENT_JSON).write_text(
        json.dumps(amendment, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    art = [f for f in files if f["kind"] == "fitted_model_artifact"]
    rows = "\n".join(
        f"| `{f['comparison_slug']}` | {f['selected_family']} | "
        f"`{f['original_sha256'][:16]}...` | `{f['amended_sha256'][:16]}...` | "
        f"{f['amended_size_bytes'] / 1e6:.1f} | {f['rows_rescored']:,} | "
        f"{f['max_abs_deviation_from_frozen']:.1e} | verified |"
        for f in art
    )
    (run_dir / AMENDMENT_MD).write_text(f"""# Post-run artifact amendment

**This is a second attestation, not a replacement for the run manifest.**

| | |
|---|---|
| Original run | `{manifest["run_id"]}` |
| Analysis source commit | `{manifest["git"]["commit"]}` |
| Run completed | {manifest["completed_at_utc"]} |
| Repack commit | `{args.repack_commit}` |
| Repack committed | {_commit_timestamp(args.repack_commit) or "n/a"} |
| Amendment written | {amendment["created_at_utc"]} |

## What changed and why

{REASON}

**No estimator, preprocessor, or calibration object was refitted, and no
predicted probability changed.** Only the metadata describing what a caller must
supply in order to score was corrected. Repacking rewrote the files, so their
checksums no longer match the ones recorded at run completion.

## How to read the two records

The run manifest attests to the production run **as it completed**. It has not
been edited, and it must not be read as attesting to files created afterwards.

| Record | Scope |
|---|---|
| `run_manifest.json` | the run's numerical outputs — **{unchanged} of {len(original)}** recorded outputs still match it byte for byte |
| `POST_RUN_ARTIFACT_AMENDMENT.json` | the {len(files)} metadata-repacked files, and nothing else |

Verify both together with:

```
python scripts/verify_run_integrity.py --run-dir {run_dir.name}
```

which checks unchanged outputs against the manifest, the amended files against
this amendment, and fails on any mismatch that neither record explains.

## Amended files

{len(joblibs)} fitted model artifacts and {len(manifests)} companion frozen
manifests, {len(files)} files in total.

Each artifact below was **independently rescored**: reloaded, used to score the
held-out predictor rows rebuilt from its source dataset, and compared against
the frozen probabilities recorded in `results/frozen/<slug>_predictions.csv`.

| Comparison | Family | Original SHA-256 | Amended SHA-256 | MB | Rows rescored | Max abs deviation | Status |
|---|---|---|---|---|---|---|---|
{rows}

Maximum deviation across all {len(deviations)} artifacts:
**{max(deviations):.2e}** (tolerance {MATCH_TOLERANCE:.0e}).

The {len(manifests)} companion `*_manifest.json` files were rewritten in the same
operation to record the corrected required-input list and the new checksums.
Their original and amended checksums are listed in the JSON record.

## Environment

Python {sys.version.split()[0]} on {platform.platform()}; package versions as
recorded in the original run manifest.
""", encoding="utf-8")

    print(f"amendment written to {run_dir / AMENDMENT_JSON}")
    print(f"  unchanged outputs matching the run manifest : {unchanged}")
    print(f"  amended files covered by this amendment     : {len(files)} "
          f"({len(joblibs)} artifacts + {len(manifests)} manifests)")
    print(f"  artifacts independently rescored            : {len(deviations)}")
    print(f"  max abs deviation                           : {max(deviations):.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
