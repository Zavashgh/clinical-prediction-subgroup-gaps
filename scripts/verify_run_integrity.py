"""Verify a production run against its manifest and its post-run amendment.

A completed run is attested by two records once any file has been legitimately
amended after completion:

    run_manifest.json                  the run's outputs at completion
    POST_RUN_ARTIFACT_AMENDMENT.json   files amended afterwards, and only those

This checks each file against whichever record claims it, and fails on any
mismatch that neither record explains. A file whose checksum has drifted with no
amendment covering it is a failure, not a warning: that is the case this script
exists to catch.

Usage:
    python scripts/verify_run_integrity.py --run-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_JSON = "POST_RUN_ARTIFACT_AMENDMENT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    staging = run_dir / "staging"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    recorded = {o["path"]: o for o in manifest["outputs"]}

    amendment_path = run_dir / AMENDMENT_JSON
    amendment = None
    amended = {}
    if amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        amended = {f["path"]: f for f in amendment["files"]}

    matched_manifest = []
    matched_amendment = []
    unexplained = []
    missing = []

    for path, record in sorted(recorded.items()):
        current = staging / path
        if not current.exists():
            missing.append(path)
            continue
        digest = sha256(current)

        if digest == record["sha256"]:
            # Still exactly what the run produced. An amendment must not claim
            # a file that never actually changed.
            if path in amended:
                unexplained.append(
                    (path, "amendment claims a file that still matches the manifest")
                )
            else:
                matched_manifest.append(path)
            continue

        entry = amended.get(path)
        if entry is None:
            unexplained.append((path, "checksum differs and no amendment covers it"))
        elif entry["amended_sha256"] != digest:
            unexplained.append(
                (path, "checksum matches neither the manifest nor the amendment")
            )
        elif entry["original_sha256"] != record["sha256"]:
            unexplained.append(
                (path, "amendment records a different original checksum")
            )
        else:
            matched_amendment.append(path)

    # An amendment may only cover files the run actually recorded.
    for path in amended:
        if path not in recorded:
            unexplained.append((path, "amendment covers a file the run never recorded"))

    if not args.quiet:
        print(f"run              : {manifest['run_id']}")
        print(f"recorded outputs : {len(recorded)}")
        print(f"  match original run manifest : {len(matched_manifest)}")
        print(f"  match amendment record      : {len(matched_amendment)}")
        print(f"  unexplained mismatches      : {len(unexplained)}")
        if missing:
            print(f"  MISSING                     : {len(missing)}")
        if amendment:
            block = amendment["amendment"]
            print(f"\namendment: {block['amended_model_artifacts']} artifacts + "
                  f"{block['amended_frozen_manifests']} manifests, "
                  f"refit={block['refit_performed']}, "
                  f"max deviation={block['max_abs_deviation_across_artifacts']:.2e}")
            print(f"  reason: {block['reason'][:78]}...")
        for path, why in unexplained[:20]:
            print(f"  UNEXPLAINED {path}: {why}")
        for path in missing[:20]:
            print(f"  MISSING {path}")

    return 1 if (unexplained or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
