"""Regression tests for field-specific JAMA replay acceptance rules."""

import json
import math
from pathlib import Path
import unittest

from src import reproducibility_policy as policy


class ReproducibilityPolicyTests(unittest.TestCase):
    def test_exact_fields_do_not_receive_numeric_tolerance(self):
        for field in (
            "selected_model_family",
            "operating_threshold",
            "subgroup_and_event_counts",
            "direction_and_significance",
            "schemas_and_output_paths",
        ):
            self.assertTrue(policy.field_values_accepted(field, "same", "same"))
            self.assertFalse(policy.field_values_accepted(field, "first", "second"))

    def test_manuscript_numerics_use_strict_absolute_tolerance(self):
        field = "manuscript_metrics_and_bootstrap_confidence_intervals"
        self.assertTrue(policy.field_values_accepted(field, 0.25, 0.25 + 1e-12))
        self.assertFalse(policy.field_values_accepted(field, 0.25, 0.25 + 2e-12))

    def test_continuous_diagnostics_use_approved_absolute_tolerance(self):
        field = "continuous_auroc_and_calibrated_score_diagnostics"
        self.assertTrue(policy.field_values_accepted(field, 0.8, 0.8 + 1e-5))
        self.assertFalse(policy.field_values_accepted(field, 0.8, 0.8 + 1.1e-5))

    def test_continuous_diagnostic_requires_all_downstream_gates(self):
        unchanged = {
            "selected_family_exact": True,
            "threshold_exact": True,
            "manuscript_outputs_accepted": True,
            "direction_exact": True,
            "significance_exact": True,
        }
        self.assertTrue(policy.continuous_diagnostic_replay_accepted(
            0.806602, 0.806607, **unchanged
        ))
        changed = dict(unchanged)
        changed["threshold_exact"] = False
        self.assertFalse(policy.continuous_diagnostic_replay_accepted(
            0.806602, 0.806607, **changed
        ))

    def test_relative_tolerance_is_zero_even_for_large_values(self):
        self.assertEqual(policy.REPRODUCIBILITY_RTOL, 0.0)
        self.assertFalse(policy.numeric_within_absolute_tolerance(
            1_000_000.0, 1_000_000.00002, atol=1e-5
        ))
        self.assertFalse(policy.numeric_within_absolute_tolerance(
            math.inf, math.inf, atol=1e-5
        ))

    def test_manifest_policy_is_complete_and_json_safe(self):
        manifest = policy.manifest_reproducibility_policy()
        self.assertEqual(
            manifest["profile"],
            "fixed_eight_thread_tolerance_based_not_bitwise",
        )
        self.assertEqual(manifest["relative_tolerance"], 0.0)
        self.assertEqual(set(manifest["field_rules"]), set(policy.FIELD_RULES))

    def test_focused_cchs_evidence_satisfies_conditional_acceptance(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "audit"
            / "cchs_eight_thread_reproducibility_diagnostic_2026-07-21.json"
        )
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["repetitions"], 5)
        self.assertEqual(
            set(evidence["selected_family_by_repetition"]),
            {"calibrated_ensemble"},
        )
        self.assertEqual(
            len(set(evidence["operating_threshold_by_repetition"])), 1
        )
        self.assertTrue(evidence["thresholded_counts_and_metrics_exact_across_repetitions"])
        self.assertTrue(evidence["within_approved_auroc_absolute_tolerance_1e_5"])
        self.assertLessEqual(
            evidence["maximum_pairwise_test_auroc_difference"],
            policy.CONTINUOUS_DIAGNOSTIC_ATOL,
        )
        self.assertLessEqual(
            evidence["successful_run_comparison"][
                "maximum_observed_cchs_auroc_difference"
            ],
            policy.CONTINUOUS_DIAGNOSTIC_ATOL,
        )
        self.assertFalse(evidence["production_outputs_written"])


if __name__ == "__main__":
    unittest.main()
