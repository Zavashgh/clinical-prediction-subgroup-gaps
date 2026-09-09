"""
figures.py
==========
Plotting helpers shared across notebooks. Each function returns the
matplotlib Figure so notebooks can further tweak or save it
(e.g. `fig.savefig("results/fig3_raw_gaps.png", dpi=150, bbox_inches="tight")`).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_curve(y_true, y_prob, group, group_labels, n_bins=10, title=None):
    """
    Reliability diagram: predicted probability (x) vs. observed frequency (y),
    one line per subgroup, plus the diagonal "perfectly calibrated" line.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for val, label in group_labels.items():
        mask = group == val
        if mask.sum() == 0:
            continue
        bin_idx = np.clip(np.digitize(y_prob[mask], bins) - 1, 0, n_bins - 1)
        obs_freq = np.full(n_bins, np.nan)
        for b in range(n_bins):
            in_bin = bin_idx == b
            if in_bin.sum() > 0:
                obs_freq[b] = y_true[mask][in_bin].mean()
        ax.plot(bin_centers, obs_freq, marker="o", label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Calibration by subgroup")
    ax.legend()
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
