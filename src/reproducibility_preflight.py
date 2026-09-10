"""Fast synthetic reproducibility check for the fixed eight-thread profile."""

import json
import os

if os.environ.get("MEDICAL_FAIRNESS_N_JOBS") != "8":
    raise RuntimeError("JAMA reproducibility preflight requires MEDICAL_FAIRNESS_N_JOBS=8")

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .pipeline import _build_models
from .reproducibility_policy import (
    CONTINUOUS_DIAGNOSTIC_ATOL,
    REPRODUCIBILITY_RTOL,
)
from .runtime import DETERMINISTIC_N_JOBS, THREAD_ENVIRONMENT


def _execution():
    X, y = make_classification(
        n_samples=320, n_features=12, n_informative=7, n_redundant=2,
        class_sep=0.8, random_state=42,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    models = _build_models(42)
    for name, model in models.items():
        values = cross_val_score(
            model, X, y, cv=folds, scoring="roc_auc", n_jobs=DETERMINISTIC_N_JOBS
        )
        scores[name] = values
    selected = max(scores, key=lambda name: float(np.mean(scores[name])))
    fitted = clone(models[selected]).fit(X, y)
    probabilities = fitted.predict_proba(X)[:, 1]
    return selected, scores, probabilities


def main():
    if DETERMINISTIC_N_JOBS != 8 or set(THREAD_ENVIRONMENT.values()) != {"8"}:
        raise RuntimeError("The fixed eight-thread environment is not active")
    first = _execution()
    second = _execution()
    same_family = first[0] == second[0]
    arrays = [
        (first[1][name], second[1][name]) for name in first[1]
    ] + [(first[2], second[2])]
    exact = same_family and all(np.array_equal(a, b) for a, b in arrays)
    max_abs = max(float(np.max(np.abs(a - b))) for a, b in arrays)
    within_tolerance = same_family and all(
        np.allclose(
            a,
            b,
            rtol=REPRODUCIBILITY_RTOL,
            atol=CONTINUOUS_DIAGNOSTIC_ATOL,
        )
        for a, b in arrays
    )
    if not within_tolerance:
        raise RuntimeError("Eight-thread synthetic executions were not reproducible")
    print(json.dumps({
        "thread_count": 8,
        "selected_family_first": first[0],
        "selected_family_second": second[0],
        "reproducibility": "exact" if exact else "tolerance_based",
        "rtol": REPRODUCIBILITY_RTOL,
        "continuous_diagnostic_atol": CONTINUOUS_DIAGNOSTIC_ATOL,
        "maximum_absolute_difference": max_abs,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
