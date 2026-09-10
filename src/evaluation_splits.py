"""Shared helpers for contamination-free auxiliary evaluation partitions."""

import pandas as pd
from sklearn.model_selection import train_test_split


def isolated_nonprimary_holdout_indices(
    df,
    target_col,
    group_col,
    groups,
    test_size=0.3,
    random_state=42,
):
    """Create reproducible holdouts for groups excluded from model fitting."""
    test_parts = []
    for group_value in groups:
        subgroup = df[df[group_col] == group_value]
        if len(subgroup) < 2:
            raise ValueError(
                f"Cannot create an isolated holdout for {group_col}={group_value!r}: "
                f"only {len(subgroup)} row(s)"
            )
        stratify = None
        outcome_counts = subgroup[target_col].value_counts()
        if len(outcome_counts) == 2 and outcome_counts.min() >= 2:
            stratify = subgroup[target_col]
        _, test_index = train_test_split(
            subgroup.index,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        test_parts.append(pd.Index(test_index))

    if not test_parts:
        return pd.Index([], dtype=int)
    combined = test_parts[0]
    for part in test_parts[1:]:
        combined = combined.append(part)
    if combined.has_duplicates:
        raise RuntimeError("Auxiliary holdout contains duplicate row indices")
    return combined


def assert_no_training_overlap(training_index, evaluation_index, context="auxiliary"):
    """Raise a controlled error unless model-training and evaluation rows are disjoint."""
    training_index = pd.Index(training_index)
    evaluation_index = pd.Index(evaluation_index)
    overlap = training_index.intersection(evaluation_index)
    if len(overlap):
        raise RuntimeError(
            f"{context}: {len(overlap)} auxiliary evaluation row(s) overlap model training"
        )
    return True


def assert_no_training_cluster_overlap(
    training_clusters, evaluation_clusters, context="auxiliary"
):
    """Raise unless the auxiliary evaluation shares no cluster with training.

    Row-level disjointness is not sufficient once a clustering identifier
    exists: a non-primary-band row can belong to a patient whose other
    encounters trained the model. This enforces the patient-level version of
    the same guarantee.
    """
    overlap = set(training_clusters) & set(evaluation_clusters)
    if overlap:
        raise RuntimeError(
            f"{context}: {len(overlap)} cluster(s) appear in both model "
            "training and the auxiliary evaluation set"
        )
    return True
