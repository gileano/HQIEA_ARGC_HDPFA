"""
Figures for the PDP 2027 paper (submission2/). Run from the repo root:
    python submission2/make_figures.py

trace_restart.pdf -- hypervolume vs generation, QIEA with/without restart and the four
baselines, from results/trace_restart_<instance>_seed<k>.csv
(src/diag_qiea_restart_trace.py). Each seed's curves are divided by the best final HV
of any algorithm in that seed (each CSV has its own reference point, so raw HV is not
comparable across seeds), then the median over seeds is plotted.

Colors: reference categorical palette slots 1-5 in fixed order (dataviz skill); both
QIEA variants share slot 1 and differ by line style. Every series also has its own
line style + marker so the figure survives grayscale printing and CVD.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

INSTANCES = ["A-n32-k5", "route2_199", "route1_334"]
SERIES = [  # (column, label, color, linestyle, marker)
    ("QIEA", "HQIEA-ARGC", "#2a78d6", "-", "o"),
    ("QIEA-norestart", "HQIEA-ARGC, no restart", "#2a78d6", "--", "x"),
    ("NSGA-II", "NSGA-II", "#eb6834", "-", "s"),
    ("SPEA2", "SPEA2", "#1baf7a", "-.", "^"),
    ("MOEA/D", "MOEA/D", "#eda100", ":", "D"),
    ("RVEA", "RVEA", "#e87ba4", (0, (5, 1, 1, 1)), "v"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(instance):
    files = sorted(RESULTS.glob(f"trace_restart_{instance}_seed*.csv"))
    normed = []
    for f in files:
        df = pd.read_csv(f).groupby("gen").last()  # gen 0 may appear twice
        df = df[df.index > 0]
        best_final = df.iloc[-1].max()
        normed.append(df / best_final)
    stacked = np.stack([d.values for d in normed])
    return pd.DataFrame(np.median(stacked, axis=0), index=normed[0].index,
                        columns=normed[0].columns), len(files)


def trace_figure():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), sharey=True)
    for ax, inst in zip(axes, INSTANCES):
        df, n_seeds = load(inst)
        for col, label, color, ls, marker in SERIES:
            ax.plot(df.index, df[col], color=color, linestyle=ls, linewidth=1.3,
                    marker=marker, markersize=3.5, markevery=5, label=label)
        ax.set_title(f"{inst} ({n_seeds} seeds)", fontsize=8, color=INK)
        ax.set_xlabel("Generation")
        ax.set_xlim(0, 500)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("HV / best final HV")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=7, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "trace_restart.pdf", bbox_inches="tight")
    fig.savefig(OUT / "trace_restart.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'trace_restart.pdf'}")


if __name__ == "__main__":
    trace_figure()
