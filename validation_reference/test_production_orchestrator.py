"""Synthetic tests for staged production orchestration and provenance."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from src import production


COMMIT = "a" * 40
GIT = {
    "commit": COMMIT,
    "branch": "main",
    "dirty": False,
    "tracked_status": [],
}


class ProductionOrchestratorTests(unittest.TestCase):
    def _layout(self, root: Path, name: str = "synthetic_run"):
        return production.create_run_layout(root, COMMIT, run_id=name)

    def test_automatic_provenance_path_and_staging_setup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = self._layout(root)
            environment = production.build_command_environment(
                layout, GIT, [], "synthetic_analysis", base_environment={}
            )
            self.assertEqual(
                Path(environment["MEDICAL_FAIRNESS_PROVENANCE_JSONL"]),
                layout.provenance_jsonl.resolve(),
            )
            self.assertEqual(
                Path(environment["MEDICAL_FAIRNESS_RESULTS_DIR"]),
                layout.staging_results.resolve(),
            )
            self.assertEqual(
                environment["MEDICAL_FAIRNESS_COMMAND_ID"], "synthetic_analysis"
            )
            self.assertEqual(environment["MEDICAL_FAIRNESS_FAIL_FAST"], "1")
            self.assertNotEqual(layout.staging_results, root / "results")

    def test_manifest_records_field_specific_reproducibility_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = self._layout(Path(temp_dir))
            manifest = production.initial_manifest(layout, GIT, [])
        self.assertEqual(manifest["schema_version"], "2.2.0")
        policy = manifest["protocol"]["reproducibility_acceptance"]
        self.assertEqual(
            policy["profile"],
            "fixed_eight_thread_tolerance_based_not_bitwise",
        )
        self.assertEqual(policy["relative_tolerance"], 0.0)
        continuous = policy["field_rules"][
            "continuous_auroc_and_calibrated_score_diagnostics"
        ]
        self.assertEqual(continuous["atol"], 1e-5)
        self.assertEqual(continuous["rtol"], 0.0)

    def test_deterministic_environment_precedes_project_imports(self):
        root = Path(production.__file__).resolve().parents[1]
        entry = root / "scripts" / "run_production.py"
        tree = ast.parse(entry.read_text(encoding="utf-8"))
        project_import_line = min(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "src.production"
        )
        thread_loop_line = min(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Attribute)
                and isinstance(child.value.value, ast.Name)
                and child.value.value.id == "os"
                and child.value.attr == "environ"
                for child in ast.walk(node)
            )
        )
        self.assertLess(thread_loop_line, project_import_line)
        source_tree = ast.parse(Path(production.__file__).read_text(encoding="utf-8"))
        forbidden = {
            "numpy", "pandas", "scipy", "sklearn", "statsmodels", "xgboost"
        }
        imported = set()
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(imported))

    def test_output_hash_finalization_and_strict_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir) / "results"
            results.mkdir()
            output = results / "example.csv"
            output.write_bytes(b"a,b\n1,2\n")
            records = production.finalize_output_files(
                results, ["example.csv"], "synthetic"
            )
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["path"], "results/example.csv")
            self.assertEqual(
                record["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            self.assertEqual(record["size_bytes"], output.stat().st_size)
            self.assertEqual(record["file_type"], "csv")
            self.assertEqual(record["producing_command_id"], "synthetic")
            with self.assertRaises(ValueError):
                production._strict_json_text({"invalid": float("nan")})

    def test_pipeline_record_is_associated_only_with_producing_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provenance.jsonl"
            records = [
                {
                    "record_type": "pipeline_analysis",
                    "command_id": "first",
                    "outputs": [],
                    "outputs_pending_command_finalization": True,
                },
                {
                    "record_type": "pipeline_analysis",
                    "command_id": "second",
                    "outputs": [],
                    "outputs_pending_command_finalization": True,
                },
            ]
            production._write_jsonl(path, records)
            output = {
                "path": "results/first.csv",
                "sha256": "b" * 64,
                "size_bytes": 1,
                "file_type": "csv",
                "media_type": "text/csv",
                "finalized_at_utc": production.utc_now(),
                "producing_command_id": "first",
                "disposition": "generated",
            }
            count = production.finalize_pipeline_records(
                path, "first", [output], require_records=True
            )
            self.assertEqual(count, 1)
            finalized = production._read_jsonl(path)
            self.assertFalse(finalized[0]["outputs_pending_command_finalization"])
            self.assertEqual(finalized[0]["outputs"], [output])
            self.assertTrue(finalized[1]["outputs_pending_command_finalization"])

    def test_failed_run_preserves_partial_provenance_and_names_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "results"
            live.mkdir()
            (live / "protected.csv").write_bytes(b"original")
            layout = self._layout(root)
            spec = production.CommandSpec(
                "failing_analysis", ("synthetic",), ("required.csv",)
            )

            def fail_executor(spec, environment, cwd, stdout, stderr):
                staged = Path(environment["MEDICAL_FAIRNESS_RESULTS_DIR"])
                (staged / "partial.csv").write_bytes(b"partial")
                stdout.write_text("partial output", encoding="utf-8")
                stderr.write_text("controlled failure", encoding="utf-8")
                return 9

            manifest = production.execute_production_plan(
                layout,
                GIT,
                [],
                [spec],
                executor=fail_executor,
                include_retained_outputs=False,
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_analysis"], "failing_analysis")
            self.assertFalse(manifest["promotion"]["eligible"])
            self.assertEqual(
                manifest["commands"][0]["outputs"][0]["path"],
                "results/partial.csv",
            )
            records = production._read_jsonl(layout.provenance_jsonl)
            self.assertEqual(records[-1]["state"], "failed")
            self.assertEqual(
                records[-1]["execution"]["n_jobs"], production.CONFIGURED_N_JOBS
            )
            self.assertEqual(
                records[-1]["execution"]["thread_environment"],
                production.THREAD_ENVIRONMENT,
            )
            self.assertEqual((live / "protected.csv").read_bytes(), b"original")

    def test_incomplete_checkpoint_is_not_promotion_eligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = self._layout(root)
            manifest = production.initial_manifest(layout, GIT, [])
            production.checkpoint_manifest(layout, manifest)
            saved = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "incomplete")
            self.assertIsNone(saved["completed_at_utc"])
            self.assertEqual(
                saved["integrity"]["manifest_payload_sha256"],
                production._manifest_payload_hash(saved),
            )
            sidecar_hash = layout.run_manifest_sha256.read_text(
                encoding="utf-8"
            ).split()[0]
            self.assertEqual(sidecar_hash, production.sha256_file(layout.run_manifest))
            with self.assertRaisesRegex(production.ProductionError, "not complete"):
                production.validate_promotion_eligibility(layout.run_dir)

    def test_notebook_staging_rewrites_copy_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.ipynb"
            original = {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": [
                            "from pathlib import Path\n",
                            "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / \"src\").is_dir())\n",
                            "RESULTS_DIR = ROOT / \"results\"\n",
                        ],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            source.write_text(json.dumps(original), encoding="utf-8")
            before = source.read_bytes()
            destination = root / "prepared.ipynb"
            production.prepare_notebook_copy(source, destination)
            self.assertEqual(source.read_bytes(), before)
            prepared = destination.read_text(encoding="utf-8")
            self.assertIn("MEDICAL_FAIRNESS_PROJECT_ROOT", prepared)
            self.assertIn("MEDICAL_FAIRNESS_RESULTS_DIR", prepared)

    def test_every_direct_production_runner_accepts_staging_results(self):
        root = Path(production.__file__).resolve().parents[1]
        runner_paths = [
            root / "run_age_decomposition.py",
            root / "run_race_decomposition.py",
            *sorted((root / "extended_robustness_scripts").glob("*.py")),
        ]
        self.assertGreater(len(runner_paths), 2)
        for runner in runner_paths:
            self.assertIn(
                "MEDICAL_FAIRNESS_RESULTS_DIR",
                runner.read_text(encoding="utf-8"),
                runner.name,
            )

    def test_extended_runners_do_not_swallow_production_failures(self):
        root = Path(production.__file__).resolve().parents[1]
        for runner in sorted((root / "extended_robustness_scripts").glob("*.py")):
            lines = runner.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if line.lstrip().startswith("except "):
                    nearby = "\n".join(lines[index + 1 : index + 4])
                    self.assertIn(
                        "MEDICAL_FAIRNESS_FAIL_FAST",
                        nearby,
                        f"{runner.name}:{index + 1}",
                    )

    def test_production_order_respects_dependencies_and_postflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = self._layout(root)
            command_ids = [
                spec.command_id
                for spec in production.production_command_specs(root, layout)
            ]
            self.assertEqual(command_ids[0], "preflight_unit_tests")
            self.assertEqual(command_ids[1], "preflight_six_model_families")
            self.assertEqual(command_ids[2], "preflight_output_contracts")
            self.assertLess(
                command_ids.index("preflight_output_contracts"),
                command_ids.index("sex_cdc"),
            )
            contract_spec = next(
                spec
                for spec in production.production_command_specs(root, layout)
                if spec.command_id == "preflight_output_contracts"
            )
            self.assertEqual(
                contract_spec.argv[-3:], ("-B", "-m", "src.output_contracts")
            )
            self.assertEqual(contract_spec.expected_outputs, ())
            self.assertFalse(contract_spec.expects_pipeline_provenance)
            self.assertLess(
                command_ids.index("extended_48_49_50"),
                command_ids.index("extended_42_44_45"),
            )
            self.assertEqual(command_ids[-1], "postflight_unit_tests")

    def test_direct_nested_cv_runner_requires_only_command_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "results").mkdir()
            layout = self._layout(root)
            spec = next(
                item for item in production.production_command_specs(root, layout)
                if item.command_id == "extended_31_33"
            )
            self.assertFalse(spec.expects_pipeline_provenance)

            def direct_model_executor(spec, environment, cwd, stdout, stderr):
                staged = Path(environment["MEDICAL_FAIRNESS_RESULTS_DIR"])
                for relative in spec.expected_outputs:
                    path = staged / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic\n", encoding="utf-8")
                stdout.write_text("direct models completed", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                return 0

            manifest = production.execute_production_plan(
                layout, GIT, [], [spec], executor=direct_model_executor,
                include_retained_outputs=False,
            )
            self.assertEqual(manifest["status"], "complete")
            records = production._read_jsonl(layout.provenance_jsonl)
            self.assertEqual([r["record_type"] for r in records], ["production_command"])
            self.assertEqual(records[0]["pipeline_record_count"], 0)

    def test_jama_plan_is_focused_and_excludes_optional_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = self._layout(root)
            specs = production.jama_command_specs(root, layout)
            ids = [item.command_id for item in specs]
            self.assertEqual(ids, [
                "preflight_unit_tests",
                "preflight_six_model_families",
                "preflight_output_contracts",
                "preflight_eight_thread_reproducibility",
                "sex_cdc", "sex_diabetes130", "sex_brfss", "sex_nhanes", "sex_cchs",
                "age_decomposition", "race_decomposition",
                "jama_supplement_analyses", "jama_reporting_sources",
                "postflight_unit_tests",
            ])
            self.assertFalse(any(item.startswith("extended_") for item in ids))
            age = next(item for item in specs if item.command_id == "age_decomposition")
            race = next(item for item in specs if item.command_id == "race_decomposition")
            self.assertIn("--jama-only", age.argv)
            self.assertIn("--jama-only", race.argv)

    def test_jama_entrypoint_sets_eight_threads_before_project_import(self):
        root = Path(production.__file__).resolve().parents[1]
        source = (root / "scripts" / "run_production.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        import_line = min(
            node.lineno for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "src.production"
        )
        assignment_line = min(
            node.lineno for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_thread_count"
                    for target in node.targets)
        )
        self.assertLess(assignment_line, import_line)
        self.assertIn('"8" if _requested_action in {"plan-jama", "run-jama"}', source)

    def test_promotion_refuses_failed_validation_without_live_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "results"
            live.mkdir()
            protected = live / "protected.txt"
            protected.write_bytes(b"unchanged")
            layout = self._layout(root)
            manifest = production.initial_manifest(layout, GIT, [])
            manifest["status"] = "complete"
            manifest["completed_at_utc"] = production.utc_now()
            manifest["validation"]["status"] = "failed"
            production.checkpoint_manifest(layout, manifest)
            with self.assertRaisesRegex(production.ProductionError, "validation"):
                production.validate_promotion_eligibility(layout.run_dir)
            self.assertEqual(protected.read_bytes(), b"unchanged")

    def test_dirty_tracked_tree_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "synthetic@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Synthetic Test"],
                cwd=root,
                check=True,
            )
            tracked = root / "tracked.txt"
            tracked.write_text("clean", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            tracked.write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(production.ProductionError, "dirty tracked"):
                production.require_clean_tracked_tree(root)

    def test_successful_staged_run_does_not_mutate_live_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "results"
            live.mkdir()
            existing = live / "existing.csv"
            existing.write_bytes(b"live-state")
            before = hashlib.sha256(existing.read_bytes()).hexdigest()
            layout = self._layout(root)
            spec = production.CommandSpec(
                "synthetic_success", ("synthetic",), ("new.csv",)
            )

            def success_executor(spec, environment, cwd, stdout, stderr):
                staged = Path(environment["MEDICAL_FAIRNESS_RESULTS_DIR"])
                (staged / "new.csv").write_bytes(b"new-staged-output")
                stdout.write_text("ok", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                return 0

            manifest = production.execute_production_plan(
                layout,
                GIT,
                [],
                [spec],
                executor=success_executor,
                include_retained_outputs=False,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["validation"]["status"], "passed")
            self.assertTrue(manifest["validation"]["live_results_unchanged"])
            self.assertTrue(manifest["promotion"]["eligible"])
            self.assertEqual(
                hashlib.sha256(existing.read_bytes()).hexdigest(), before
            )
            self.assertFalse((live / "new.csv").exists())
            self.assertTrue((layout.staging_results / "new.csv").is_file())


if __name__ == "__main__":
    unittest.main()
