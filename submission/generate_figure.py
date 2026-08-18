"""
Generates the results figure for the abstract submission from the full-scale
30-run/n-gen=500 campaign data in results/*_indicators.csv (logs.txt section 15e).

Not part of the research pipeline proper (see src/) -- a one-off plotting
script for the abstract deliverable in submission/. Re-run after any change
to results/*_indicators.csv to keep the figure in sync.
"""
import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTANCE_ORDER = [
    "A-n32-k5", "A-n33-k5", "E-n101-k8", "M-n200-k16",
    "route1_334", "route2_199", "route3_202",
]

rows = []
for f in glob.glob("results/*_indicators.csv"):
    df = pd.read_csv(f)
    inst = df["instance"].iloc[0]
    means = df.groupby("algorithm")["hypervolume"].mean()
    best_alg = means.idxmax()
    rows.append({
        "instance": inst,
        "ratio": means["QIEA"] / means[best_alg],
        "best_alg": best_alg,
    })

tbl = pd.DataFrame(rows).set_index("instance").loc[INSTANCE_ORDER].reset_index()

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=200)

bar_color = "#4C6E9C"
tie_color = "#C7743A"
colors = [tie_color if inst == "route1_334" else bar_color for inst in tbl["instance"]]

bars = ax.bar(tbl["instance"], tbl["ratio"], color=colors, width=0.6, zorder=3)
ax.axhline(1.0, color="0.3", linewidth=0.9, linestyle="--", zorder=2)
ax.text(6.55, 1.02, "best baseline = 1.0", ha="right", va="bottom", fontsize=8, color="0.3")

for bar, (_, row) in zip(bars, tbl.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{row['ratio']:.2f}\n({row['best_alg']})", ha="center", va="bottom", fontsize=7.3)

ax.set_ylim(0, 1.18)
ax.set_ylabel("Mean hypervolume ratio\n(HQIEA-ARGC / best baseline)")
ax.set_xticklabels(tbl["instance"], rotation=20, ha="right")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, color="0.88", zorder=0)
ax.set_axisbelow(True)
ax.set_title(
    "HQIEA-ARGC vs. best classical baseline, 30 runs / 500 generations, all 7 CVRP instances",
    fontsize=9.5,
)

fig.tight_layout()
fig.savefig("submission/figures/hv_ratio_comparison.png", bbox_inches="tight")
print(tbl)
print("saved submission/figures/hv_ratio_comparison.png")
