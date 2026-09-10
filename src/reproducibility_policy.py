"""Field-specific acceptance rules for fixed-eight-thread JAMA replays.

The eight-thread profile fixes seeds, inputs, estimators, and native thread
limits, but parallel floating-point reductions are not promised to be
bitwise-identical.  Acceptance is therefore exact for decisions and discrete
outputs, and absolute-tolerance based only for explicitly named numerical
fields.  Relative tolerance is always zero so a large-valued field cannot
receive a looser comparison merely because of its magnitude.
"""

from __future__ import annotations

import math
from typing import Any


STRICT_NUMERIC_ATOL = 1e-12
CONTINUOUS_DIAGNOSTIC_ATOL = 1e-5
REPRODUCIBILITY_RTOL = 0.0

FIELD_RULES = {
    "selected_model_family": {
        "comparison": "exact",
    },
    "operating_threshold": {
        "comparison": "exact",
    },
    "subgroup_and_event_counts": {
        "comparison": "exact",
    },
    "manuscript_metrics_and_bootstrap_confidence_intervals": {
        "comparison": "exact_then_absolute_tolerance",
        "atol": STRICT_NUMERIC_ATOL,
        "rtol": REPRODUCIBILITY_RTOL,
    },
    "direction_and_significance": {
        "comparison": "exact",
    },
    "schemas_and_output_paths": {
        "comparison": "exact",
    },
    "continuous_auroc_and_calibrated_score_diagnostics": {
        "comparison": "absolute_tolerance",
        "atol": CONTINUOUS_DIAGNOSTIC_ATOL,
        "rtol": REPRODUCIBILITY_RTOL,
        "condition": "no material downstream change",
    },
}


def numeric_within_absolute_tolerance(
    first: float,
    second: float,
    *,
    atol: float,
) -> bool:
    """Compare finite numerical values with an absolute tolerance only."""
    first_value = float(first)
    second_value = float(second)
    if not (math.isfinite(first_value) and math.isfinite(second_value)):
        return False
    return abs(first_value - second_value) <= float(atol)


def field_values_accepted(field_class: str, first: Any, second: Any) -> bool:
    """Apply the documented acceptance rule for one field class."""
    try:
        rule = FIELD_RULES[field_class]
    except KeyError as exc:
        raise ValueError(f"Unknown reproducibility field class: {field_class}") from exc
    if rule["comparison"] == "exact":
        return first == second
    return numeric_within_absolute_tolerance(
        first,
        second,
        atol=float(rule["atol"]),
    )


def continuous_diagnostic_replay_accepted(
    first: float,
    second: float,
    *,
    selected_family_exact: bool,
    threshold_exact: bool,
    manuscript_outputs_accepted: bool,
    direction_exact: bool,
    significance_exact: bool,
) -> bool:
    """Accept a continuous diagnostic only when all downstream gates hold."""
    downstream_unchanged = all((
        selected_family_exact,
        threshold_exact,
        manuscript_outputs_accepted,
        direction_exact,
        significance_exact,
    ))
    return downstream_unchanged and field_values_accepted(
        "continuous_auroc_and_calibrated_score_diagnostics",
        first,
        second,
    )


def manifest_reproducibility_policy() -> dict[str, object]:
    """Return a JSON-safe copy suitable for run-level provenance."""
    return {
        "profile": "fixed_eight_thread_tolerance_based_not_bitwise",
        "relative_tolerance": REPRODUCIBILITY_RTOL,
        "field_rules": {
            name: dict(rule)
            for name, rule in FIELD_RULES.items()
        },
    }
