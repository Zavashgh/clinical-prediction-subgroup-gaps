"""Training-fitted preprocessing for the corrected analysis path.

Rationale
---------
The original loaders performed median imputation and one-hot construction on
the full dataset before the comparison-specific train/test split. Test-row
predictor values therefore contributed to the imputation constants written
into training rows, and the test rows co-determined which dummy columns
existed. That is predictor-only (unsupervised) preprocessing leakage: no
outcome information crossed the split, but training feature values were not
independent of the evaluation partition.

This module confines every data-derived preprocessing decision to a single
object that is fitted on training rows only and then applied unchanged to
both partitions.

Fixed deterministic recoding (hard-coded binary/ordinal maps, age-band
midpoints, fixed BMI rescaling, medication dose maps, exclusion logic that
does not consult learned predictor statistics) remains in the loaders and is
still applied before the split. Those operations depend on no observed
statistic and cannot transmit information across the partition boundary.

What is learned here, from training rows only:
  * per-column medians for numeric imputation;
  * the category set and reference level for each one-hot source column.

Guarantees
----------
* ``fit`` never reads a row outside the training index it is given.
* ``transform`` produces the identical ordered column list for any input.
* A category present only in the test partition never creates a new feature
  column; such rows fall back to the training reference level (all-zero
  indicators) and are counted in ``unseen_category_counts_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrainFittedPreprocessor:
    """Median imputation and one-hot schema learned from training rows only.

    Parameters
    ----------
    feature_order : list of str
        The design-matrix column order, as declared by the loader. Entries
        naming a categorical source are expansion points: that source's
        learned indicator columns are inserted at exactly that position. All
        other entries are numeric/ordinal columns, median-imputed with a
        training median when they carry missing values. Declaring the source
        in place preserves the column ordering the original loaders produced,
        so the corrected run differs from the original only by the two
        intended methodological fixes and not by feature ordering.
    categorical_specs : list of dict
        Each entry is ``{"source": <column>, "prefix": <str>,
        "drop_first": <bool>}``. Categories are discovered from the training
        rows, sorted lexicographically for determinism, and (when
        ``drop_first``) the first sorted category becomes the reference level
        and gets no indicator column. This reproduces the reference-level
        semantics of ``pandas.get_dummies(..., drop_first=True)`` while
        binding the category set to the training partition.
    """

    def __init__(self, feature_order, categorical_specs=None):
        self.feature_order = list(feature_order)
        self.categorical_specs = [dict(spec) for spec in (categorical_specs or [])]
        sources = {spec["source"] for spec in self.categorical_specs}
        self.numeric_cols = [c for c in self.feature_order if c not in sources]
        missing = sources - set(self.feature_order)
        if missing:
            raise ValueError(
                "Categorical source(s) absent from feature_order, so their "
                f"position in the design matrix is undefined: {sorted(missing)}"
            )
        self.medians_ = {}
        self.categories_ = {}
        self.reference_level_ = {}
        self.dummy_columns_ = {}
        self.feature_names_ = []
        self.unseen_category_counts_ = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, frame):
        """Learn imputation medians and the one-hot schema from ``frame``.

        ``frame`` must contain training rows only. Nothing else is consulted.
        """
        self.medians_ = {}
        for col in self.numeric_cols:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.isnull().any():
                median = series.median()
                if not np.isfinite(median):
                    raise ValueError(
                        f"Training median for {col!r} is not finite; the column "
                        "is empty or entirely missing in the training partition"
                    )
                self.medians_[col] = float(median)

        self.categories_ = {}
        self.reference_level_ = {}
        self.dummy_columns_ = {}
        for spec in self.categorical_specs:
            source = spec["source"]
            prefix = spec.get("prefix", source)
            observed = frame[source].dropna().unique().tolist()
            categories = sorted(observed, key=lambda value: str(value))
            if not categories:
                raise ValueError(
                    f"Categorical source {source!r} has no observed category in "
                    "the training partition"
                )
            if spec.get("drop_first", True):
                reference, kept = categories[0], categories[1:]
            else:
                reference, kept = None, categories
            self.categories_[source] = categories
            self.reference_level_[source] = reference
            self.dummy_columns_[source] = [f"{prefix}_{value}" for value in kept]

        # Expand in the loader-declared order so a categorical source's
        # indicators appear exactly where the source name sat.
        names = []
        for entry in self.feature_order:
            if entry in self.dummy_columns_:
                names.extend(self.dummy_columns_[entry])
            else:
                names.append(entry)
        self.feature_names_ = names
        self.unseen_category_counts_ = {}
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def transform(self, frame, partition="unnamed"):
        """Apply the learned preprocessing. Returns a float DataFrame.

        Columns are always ``feature_names_``, in that exact order.
        """
        if not self._fitted:
            raise RuntimeError("TrainFittedPreprocessor.transform before fit")

        pieces = {}
        for col in self.numeric_cols:
            series = pd.to_numeric(frame[col], errors="coerce")
            if col in self.medians_:
                series = series.fillna(self.medians_[col])
            pieces[col] = series.astype(float)

        for spec in self.categorical_specs:
            source = spec["source"]
            values = frame[source]
            known = self.categories_[source]
            unseen_mask = values.notna() & ~values.isin(known)
            n_unseen = int(unseen_mask.sum())
            if n_unseen:
                self.unseen_category_counts_.setdefault(partition, {})[source] = {
                    "n_rows": n_unseen,
                    "levels": sorted(
                        {str(v) for v in values[unseen_mask].unique()}
                    ),
                }
            prefix = spec.get("prefix", source)
            for category in known:
                column = f"{prefix}_{category}"
                if column not in self.dummy_columns_[source]:
                    continue  # reference level: no indicator column
                pieces[column] = (values == category).astype(float)

        out = pd.DataFrame(pieces, index=frame.index)
        out = out.reindex(columns=self.feature_names_)
        if out.isnull().to_numpy().any():
            offending = out.columns[out.isnull().any()].tolist()
            raise ValueError(
                f"Missing values remain after preprocessing in: {offending}"
            )
        return out.astype(float)

    def fit_transform(self, frame, partition="train"):
        return self.fit(frame).transform(frame, partition=partition)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def expand_columns(self, names):
        """Map block definitions onto design-matrix columns.

        A name that is a categorical source expands to its learned indicator
        columns; every other name passes through unchanged. This lets the
        case-mix covariate blocks keep referring to ``"race"`` rather than to
        a hard-coded list of dummy column names.
        """
        sources = {spec["source"] for spec in self.categorical_specs}
        expanded = []
        for name in names:
            if name in sources:
                expanded.extend(self.dummy_columns_[name])
            else:
                expanded.append(name)
        return expanded

    def schema(self):
        """Serializable record of everything this object learned."""
        return {
            "feature_order_declared": list(self.feature_order),
            "numeric_cols": list(self.numeric_cols),
            "imputed_cols": sorted(self.medians_),
            "training_medians": {k: float(v) for k, v in sorted(self.medians_.items())},
            "categorical_sources": [
                {
                    "source": spec["source"],
                    "prefix": spec.get("prefix", spec["source"]),
                    "drop_first": bool(spec.get("drop_first", True)),
                    "training_categories": [str(c) for c in self.categories_[spec["source"]]],
                    "reference_level": (
                        None
                        if self.reference_level_[spec["source"]] is None
                        else str(self.reference_level_[spec["source"]])
                    ),
                    "indicator_columns": list(self.dummy_columns_[spec["source"]]),
                }
                for spec in self.categorical_specs
            ],
            "feature_names_in_order": list(self.feature_names_),
            "n_features": len(self.feature_names_),
            "unseen_categories_by_partition": self.unseen_category_counts_,
        }
