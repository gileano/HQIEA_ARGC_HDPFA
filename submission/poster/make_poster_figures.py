"""
Figures and the results table for the EQTC 2026 A0 poster (submission/poster/). Run
from the repo root:
    python submission/poster/make_poster_figures.py

Writes into submission/poster/figures/:
  trace.pdf          HV vs generation, QIEA with/without restart + the four baselines,
                     from results/trace_restart_<instance>_seed<k>.csv
                     (src/diag_qiea_restart_trace.py). Same normalisation as
                     submission2/make_figures.py: each seed's curves are divided by the
                     best final HV of any algorithm in that seed, median over seeds.
  restart_ratio.pdf  QIEA's mean-HV ratio to the best baseline per instance, before vs
                     after the plateau-gated restart. "After" is computed from
                     results/*_indicators.csv (30 runs, n-gen=500). "Before" is the
                     pre-fix campaign (logs.txt section 14b / 15e); those CSVs were
                     backed up outside the repo, so the values are taken from the log.
  table_results.tex  per-instance HV ratio of every algorithm to the best one, with the
                     restart gain from results/tune_qiea_restart_confirm_<inst>.csv
                     (20 seeds, logs.txt section 15d) and paired Wilcoxon tests
                     QIEA-vs-baseline over the 30 runs.

Figures are drawn at their physical size on the poster (one column = 14.6 in) with
Arial, so text in them matches the poster body. Colors: reference categorical palette
slots 1-5 (dataviz skill), same assignment as submission2/make_figures.py.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

INSTANCES = ["A-n32-k5", "A-n33-k5", "E-n101-k8", "M-n200-k16",
             "route1_334", "route2_199", "route3_202"]
BASELINES = ["NSGA-II", "SPEA2", "MOEA/D", "RVEA"]
TRACE_INSTANCES = ["A-n32-k5", "route2_199", "route1_334"]

# QIEA / best-baseline mean-HV ratio before the restart fix (logs.txt section 15e)
RATIO_BEFORE = {"A-n32-k5": 0.38, "A-n33-k5": 0.38, "E-n101-k8": 0.18, "M-n200-k16": 0.17,
                "route1_334": 0.45, "route2_199": 0.37, "route3_202": 0.33}

SERIES = [  # (column, label, color, linestyle, marker)
    ("QIEA", "HQIEA-ARGC", "#2a78d6", "-", "o"),
    ("QIEA-norestart", "HQIEA-ARGC, no restart", "#2a78d6", "--", "x"),
    ("NSGA-II", "NSGA-II", "#eb6834", "-", "s"),
    ("SPEA2", "SPEA2", "#1baf7a", "-.", "^"),
    ("MOEA/D", "MOEA/D", "#eda100", ":", "D"),
    ("RVEA", "RVEA", "#e87ba4", (0, (5, 1, 1, 1)), "v"),
]
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"  # blue ramp steps 450 / 250
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_WIDTH_IN = 14.6


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 22, "axes.titlesize": 22, "axes.labelsize": 22,
        "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 20,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.linewidth": 1.2, "xtick.major.width": 1.2,
        "ytick.major.width": 1.2, "pdf.fonttype": 42,
    })


def clean_axes(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_trace(instance):
    files = sorted(RESULTS.glob(f"trace_restart_{instance}_seed*.csv"))
    normed = []
    for f in files:
        df = pd.read_csv(f).groupby("gen").last()  # gen 0 may appear twice
        df = df[df.index > 0]
        normed.append(df / df.iloc[-1].max())
    stacked = np.stack([d.values for d in normed])
    return pd.DataFrame(np.median(stacked, axis=0), index=normed[0].index,
                        columns=normed[0].columns), len(files)


def trace_figure():
    fig, axes = plt.subplots(1, 3, figsize=(COL_WIDTH_IN, 5.1), sharey=True)
    for ax, inst in zip(axes, TRACE_INSTANCES):
        df, n_seeds = load_trace(inst)
        for col, label, color, ls, marker in SERIES:
            ax.plot(df.index, df[col], color=color, linestyle=ls, linewidth=3.2,
                    marker=marker, markersize=10, markevery=5, label=label)
        ax.set_title(inst, color=INK)
        ax.set_xlabel("Generation")
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
        ax.set_ylim(0, 1.05)
        clean_axes(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    axes[0].set_ylabel("HV / best final HV")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=3.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.0)
    save(fig, "trace", n_seeds)


def ratio_after():
    rows = {}
    for inst in INSTANCES:
        df = pd.read_csv(RESULTS / f"{inst}_indicators.csv")
        means = df.groupby("algorithm")["hypervolume"].mean()
        rows[inst] = means["QIEA"] / means[BASELINES].max()
    return rows


def restart_ratio_figure():
    after = ratio_after()
    x = np.arange(len(INSTANCES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 4.7))
    before_vals = [RATIO_BEFORE[i] for i in INSTANCES]
    after_vals = [after[i] for i in INSTANCES]
    ax.bar(x - w / 2, before_vals, w, color=BLUE_LIGHT, edgecolor="white", linewidth=3,
           label="Before: no restart", zorder=3)
    ax.bar(x + w / 2, after_vals, w, color=BLUE, edgecolor="white", linewidth=3,
           label="After: plateau-gated restart", zorder=3)
    for xi, (b, a) in enumerate(zip(before_vals, after_vals)):
        ax.text(xi - w / 2, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=17, color=MUTED)
        ax.text(xi + w / 2, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=17,
                color=INK, fontweight="bold")
    ax.axhline(1.0, color=MUTED, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(len(INSTANCES) - 0.5, 1.02, "best baseline = 1.0", ha="right", va="bottom",
            fontsize=18, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(INSTANCES, rotation=18, ha="right")
    ax.set_xlim(-0.6, len(INSTANCES) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("HQIEA-ARGC HV /\nbest baseline HV")
    clean_axes(ax)
    ax.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.2))
    fig.tight_layout()
    save(fig, "restart_ratio")


def save(fig, name, n_seeds=None):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    extra = f" ({n_seeds} seeds per panel)" if n_seeds else ""
    print(f"wrote {OUT / name}.pdf{extra}")


def results_table():
    lines = []
    for inst in INSTANCES:
        df = pd.read_csv(RESULTS / f"{inst}_indicators.csv")
        means = df.groupby("algorithm")["hypervolume"].mean()
        best = means.max()
        q = df[df.algorithm == "QIEA"].sort_values("run")["hypervolume"].values
        gain = pd.read_csv(RESULTS / f"tune_qiea_restart_confirm_{inst}.csv").iloc[0]
        assert gain["wilcoxon_p"] < 0.005, inst
        cells = [inst.replace("_", r"\_"), f"+{gain['pct_change']:.1f}\\%"]
        for alg in BASELINES:
            b = df[df.algorithm == alg].sort_values("run")["hypervolume"].values
            p = stats.wilcoxon(q, b).pvalue
            val = f"{means[alg] / best:.2f}"
            if means[alg] == best:
                val = rf"\textbf{{{val}}}"
            if p >= 0.05:
                val += r"$^{\approx}$"
            cells.append(val)
        cells.append(f"{means['QIEA'] / best:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    (HERE / "table_results.tex").write_text(
        "% generated by make_poster_figures.py -- do not edit by hand\n" + "\n".join(lines) + "\n")
    print(f"wrote {HERE / 'table_results.tex'}")


if __name__ == "__main__":
    style()
    trace_figure()
    restart_ratio_figure()
    results_table()
