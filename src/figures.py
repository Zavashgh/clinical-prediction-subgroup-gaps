"""
figures.py
==========
Plotting helpers shared across notebooks. Each function returns the
matplotlib Figure so notebooks can further tweak or save it
(e.g. `fig.savefig("results/fig3_raw_gaps.png", dpi=150, bbox_inches="tight")`).
"""

import numpy as np
import matplotlib.pyplot as plt


def calibration_curve_points(y_true, y_prob, n_bins=10):
    """Reliability-diagram coordinates for one sample.

    Returns ``(x, y, counts)`` where ``x`` is the MEAN PREDICTED PROBABILITY of
    the observations that actually fall in each bin, ``y`` is the observed event
    proportion in that bin, and ``counts`` is the number of observations.

    The x-coordinate is deliberately not the nominal bin centre. Predictions are
    rarely uniform within a bin -- with isotonic calibration they often pile up
    at one edge -- so plotting against the centre shifts points horizontally
    away from where the model's predictions actually are, and makes a
    well-calibrated model look miscalibrated (or the reverse). Empty bins are
    returned as NaN rather than imputed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    x = np.full(n_bins, np.nan)
    y = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = bin_idx == b
        n = int(in_bin.sum())
        counts[b] = n
        if n:
            x[b] = y_prob[in_bin].mean()
            y[b] = y_true[in_bin].mean()
    return x, y, counts


def plot_calibration_curve(y_true, y_prob, group, group_labels, n_bins=10,
                           title=None):
    """Reliability diagram with an observation-count panel.

    One line per subgroup. Each point is placed at the mean predicted
    probability of the observations in that bin, not at the bin centre, so the
    horizontal position reflects what the model actually predicted. The lower
    panel shows how many held-out observations support each bin, because a bin
    holding a handful of observations should not be read like one holding
    thousands.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    width = (bins[1] - bins[0]) / (max(len(group_labels), 1) + 1)
    for offset, (val, label) in enumerate(group_labels.items()):
        mask = group == val
        if mask.sum() == 0:
            continue
        x, y, counts = calibration_curve_points(
            y_true[mask], y_prob[mask], n_bins=n_bins
        )
        ax.plot(x, y, marker="o", markersize=4, label=label)
        centres = (bins[:-1] + bins[1:]) / 2
        ax_n.bar(centres + (offset - 0.5) * width, counts, width=width,
                 label=label, alpha=0.85)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Perfect calibration")
    ax.set_ylabel("Observed event proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Calibration by subgroup")
    ax.legend(fontsize=8)

    ax_n.set_yscale("log")
    ax_n.set_ylabel("Observations")
    ax_n.set_xlabel("Mean predicted probability in bin")
    ax_n.grid(axis="y", linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    return fig


def plot_gap_bars(gap_dict, title=None, ylabel="Gap (A - B)"):
    """
    Bar chart comparing raw vs. adjusted gaps.
    `gap_dict` maps a label (e.g. "Raw", "Prevalence-adjusted") to a gap
    value, or to a (value, ci_low, ci_high) tuple for error bars.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(gap_dict.keys())
    values, errs_low, errs_high = [], [], []
    for v in gap_dict.values():
        if isinstance(v, tuple):
            val, lo, hi = v
            values.append(val)
            errs_low.append(val - lo)
            errs_high.append(hi - val)
        else:
            values.append(v)
            errs_low.append(0)
            errs_high.append(0)

    colors = ["#C0504D", "#2E74B5", "#70AD47", "#7030A0"]
    bars = ax.bar(labels, values, color=colors[: len(labels)],
                   yerr=[errs_low, errs_high], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Gap before vs. after adjustment")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_subgroup_bars(metrics_a, metrics_b, label_a, label_b, metric_keys, title=None):
    """Grouped bar chart comparing several metrics across two subgroups."""
    fig, ax = plt.subplots(figsize=(max(6, len(metric_keys) * 1.2), 4))
    x = np.arange(len(metric_keys))
    width = 0.35

    vals_a = [metrics_a[k] for k in metric_keys]
    vals_b = [metrics_b[k] for k in metric_keys]

    ax.bar(x - width / 2, vals_a, width, label=label_a, color="#2E74B5")
    ax.bar(x + width / 2, vals_b, width, label=label_b, color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=30, ha="right")
    ax.set_title(title or "Subgroup metric comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_forest(rows, title=None, xlabel="Gap (Male - Female)"):
    """
    Forest plot comparing a gap (with 95% CI) across multiple datasets.

    `rows` is a list of (dataset_label, point, ci_low, ci_high) tuples,
    given top-to-bottom in the order they should appear (the first row is
    drawn at the top of the plot).
    """
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]

    labels = [r[0] for r in rows]
    points = np.array([r[1] for r in rows])
    los = np.array([r[2] for r in rows])
    his = np.array([r[3] for r in rows])

    significant = (los > 0) | (his < 0)
    colors = ["#C0504D" if s else "#7F7F7F" for s in significant]

    ax.errorbar(points, y, xerr=[points - los, his - points],
                 fmt="o", capsize=4, ecolor="black", elinewidth=1,
                 markerfacecolor="none", markeredgecolor="black", zorder=2)
    ax.scatter(points, y, color=colors, zorder=3, s=50)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title or "Gap across datasets (95% CI)")
    fig.tight_layout()
    return fig


def plot_case_mix_waterfall(waterfall_df, title=None):
    """Bar chart of the group log-odds coefficient as covariate blocks are added."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(waterfall_df["step"], waterfall_df["group_coef_logodds"], color="#2E74B5")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Group coefficient (log-odds)")
    ax.set_title(title or "Case-mix waterfall: group effect on outcome")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig
