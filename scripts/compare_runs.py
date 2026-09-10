"""Exhaustive cell-by-cell comparison of two production runs' staging trees.

Rather than enumerating quantities by hand, this walks every CSV that both
runs produced and compares every cell, so no previously reported value can be
missed. It also reports files that only one run produced.

Differences are classified against the run's own recorded reproducibility
acceptance profile:

  IDENTICAL          bit-equal (or equal strings)
  WITHIN_TOLERANCE   numeric, |difference| <= the field's tolerance
  CHANGED            numeric difference beyond tolerance
  TYPE_OR_TEXT       non-numeric value differs
  STRUCTURE          schema, row count, or key set differs

The exact-tier fields (operating threshold, selected family, subgroup and
event counts, direction, significance) are held to exact equality; continuous
AUROC and calibrated-score diagnostics get the profile's 1e-5 absolute
tolerance. This script only DESCRIBES differences; it makes no judgement about
scientific importance.

Usage:
    python scripts/compare_runs.py --baseline <run_dir> --candidate <run_dir> \
        --out <report_dir>
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Absolute tolerances from the runs' recorded acceptance profile.
TOLERANCE_CONTINUOUS = 1e-5     # AUROC and calibrated-score diagnostics
TOLERANCE_EXACT = 0.0           # everything the profile marks exact

# Column-name fragments whose values are the continuous diagnostics the
# profile allows to move at 1e-5 under fixed eight-thread execution.
CONTINUOUS_HINTS = ("auroc", "brier", "calib", "ece", "auprc", "prob")

# Column-name fragments the profile holds exactly.
EXACT_HINTS = (
    "threshold", "family", "_n", "n_", "count", "numerator", "denominator",
    "tp", "fp", "tn", "fn",
)


def tolerance_for(column: str) -> float:
    name = column.lower()
    if any(hint in name for hint in CONTINUOUS_HINTS):
        return TOLERANCE_CONTINUOUS
    return TOLERANCE_EXACT


def key_columns(frame: pd.DataFrame) -> list:
    """Identify columns that jointly identify a row, for order-free alignment."""
    candidates = [
        c for c in frame.columns
        if frame[c].dtype == object or c in {"threshold", "display_order"}
    ]
    for size in range(1, min(len(candidates), 5) + 1):
        subset = candidates[:size]
        if subset and not frame.duplicated(subset=subset).any():
            return subset
    return []


def compare_frames(name, old, new, rows):
    if list(old.columns) != list(new.columns):
        only_old = [c for c in old.columns if c not in new.columns]
        only_new = [c for c in new.columns if c not in old.columns]
        rows.append({
            "file": name, "row_key": "", "column": "",
            "classification": "STRUCTURE",
            "baseline": f"columns={len(old.columns)}",
            "candidate": f"columns={len(new.columns)}",
            "difference": f"only_baseline={only_old}; only_candidate={only_new}",
        })
        shared = [c for c in old.columns if c in new.columns]
        if not shared:
            return
        old, new = old[shared], new[shared]

    keys = key_columns(old)
    if keys and set(keys).issubset(new.columns) and not new.duplicated(keys).any():
        old_i = old.set_index(keys).sort_index()
        new_i = new.set_index(keys).sort_index()
        missing = old_i.index.difference(new_i.index)
        added = new_i.index.difference(old_i.index)
        for index in list(missing)[:50]:
            rows.append({
                "file": name, "row_key": str(index), "column": "",
                "classification": "STRUCTURE", "baseline": "present",
                "candidate": "absent", "difference": "row only in baseline",
            })
        for index in list(added)[:50]:
            rows.append({
                "file": name, "row_key": str(index), "column": "",
                "classification": "STRUCTURE", "baseline": "absent",
                "candidate": "present", "difference": "row only in candidate",
            })
        common = old_i.index.intersection(new_i.index)
        old_i, new_i = old_i.loc[common], new_i.loc[common]
        labels = [str(i) for i in common]
    else:
        if len(old) != len(new):
            rows.append({
                "file": name, "row_key": "", "column": "",
                "classification": "STRUCTURE", "baseline": f"rows={len(old)}",
                "candidate": f"rows={len(new)}", "difference": "row count differs",
            })
            limit = min(len(old), len(new))
            old, new = old.iloc[:limit], new.iloc[:limit]
        old_i, new_i = old.reset_index(drop=True), new.reset_index(drop=True)
        labels = [str(i) for i in range(len(old_i))]

    for column in old_i.columns:
        a, b = old_i[column], new_i[column]
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        tol = tolerance_for(column)
        for position, label in enumerate(labels):
            x, y = a.iloc[position], b.iloc[position]
            if numeric:
                if pd.isna(x) and pd.isna(y):
                    continue
                if pd.isna(x) or pd.isna(y):
                    classification, diff = "CHANGED", "missingness differs"
                else:
                    delta = float(y) - float(x)
                    if delta == 0.0:
                        continue
                    classification = (
                        "WITHIN_TOLERANCE" if abs(delta) <= tol else "CHANGED"
                    )
                    diff = f"{delta:+.12g}"
            else:
                if (pd.isna(x) and pd.isna(y)) or str(x) == str(y):
                    continue
                classification, diff = "TYPE_OR_TEXT", "text differs"
            rows.append({
                "file": name, "row_key": label, "column": column,
                "classification": classification,
                "baseline": x, "candidate": y, "difference": diff,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.baseline) / "staging" / "results"
    cand = Path(args.candidate) / "staging" / "results"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_files = {
        p.relative_to(base).as_posix() for p in base.rglob("*.csv")
    }
    cand_files = {
        p.relative_to(cand).as_posix() for p in cand.rglob("*.csv")
    }

    rows = []
    for name in sorted(base_files - cand_files):
        rows.append({
            "file": name, "row_key": "", "column": "",
            "classification": "STRUCTURE", "baseline": "present",
            "candidate": "absent", "difference": "file only in baseline",
        })
    for name in sorted(cand_files - base_files):
        rows.append({
            "file": name, "row_key": "", "column": "",
            "classification": "NEW_OUTPUT", "baseline": "absent",
            "candidate": "present", "difference": "file only in candidate",
        })
    for name in sorted(base_files & cand_files):
        compare_frames(
            name,
            pd.read_csv(base / name),
            pd.read_csv(cand / name),
            rows,
        )

    frame = pd.DataFrame(rows, columns=[
        "file", "row_key", "column", "classification",
        "baseline", "candidate", "difference",
    ])
    frame.to_csv(out / "old_vs_new_cell_comparison.csv", index=False)

    counts = frame["classification"].value_counts().to_dict() if len(frame) else {}
    compared = sorted(base_files & cand_files)
    summary = {
        "baseline_run": Path(args.baseline).name,
        "candidate_run": Path(args.candidate).name,
        "csv_files_in_baseline": len(base_files),
        "csv_files_in_candidate": len(cand_files),
        "csv_files_compared": len(compared),
        "differences_by_classification": counts,
        "files_with_changed_cells": sorted(
            frame.loc[frame["classification"] == "CHANGED", "file"].unique().tolist()
        ),
        "new_output_files": sorted(cand_files - base_files),
        "missing_output_files": sorted(base_files - cand_files),
    }
    (out / "old_vs_new_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"CSV files compared : {len(compared)}")
    print(f"new files          : {len(cand_files - base_files)}")
    print(f"missing files      : {len(base_files - cand_files)}")
    print(f"difference rows    : {len(frame)}")
    for key, value in sorted(counts.items()):
        print(f"  {key:18s} {value}")
    if summary["files_with_changed_cells"]:
        print("\nfiles containing CHANGED cells:")
        for name in summary["files_with_changed_cells"]:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
