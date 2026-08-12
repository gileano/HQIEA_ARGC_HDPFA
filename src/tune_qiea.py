"""
OFAT hyperparameter sweep for QIEA on a cheap CVRP instance.

Purpose: qiea.py's neighborhood_size, mutation_prob, theta_max/theta_min, and
n_partitions (which sets H, the population/weight-vector count) were first-guess
defaults (paper1.txt section 7), and QIEA was the weakest or tied-weakest
algorithm on every pilot instance run so far. This sweeps one hyperparameter at
a time -- holding the rest at qiea.py's current defaults -- on A-n32-k5, the
cheapest instance, to find a config worth carrying into the full
run_experiment.py campaign.

OFAT, not full factorial: trades interaction-effect blindness for being cheap
enough to run in a few minutes instead of hours. A combined-best config,
assembled from the best value found along each dimension, is verified at the
end with more seeds against the baseline defaults.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_indicators  # noqa: E402
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

DEFAULTS = dict(neighborhood_size=10, theta_max=0.35, theta_min=0.02, mutation_prob=0.05, n_partitions=None)

SWEEP = {
    "neighborhood_size": [5, 10, 15, 20, 30, 50],
    "mutation_prob": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    "theta_max": [0.15, 0.25, 0.35, 0.5, 0.7, 0.9],
    "theta_min": [0.005, 0.01, 0.02, 0.05, 0.1],
    "n_partitions": [3, 4, 5, 6, 7, 8],
}

N_SEEDS = 5
N_GEN = 80
COMBINED_SEEDS = 10


def parse_value(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def build_configs():
    configs = [("baseline", dict(DEFAULTS))]
    for param, values in SWEEP.items():
        for v in values:
            if v == DEFAULTS[param]:
                continue
            cfg = dict(DEFAULTS)
            cfg[param] = v
            configs.append((f"{param}={v}", cfg))
    return configs


def run_config(problem, config, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(problem, decode="permutation", seed=seed, **config)
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def main(instance="A-n32-k5"):
    base = Path(__file__).resolve().parent.parent
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)

    configs = build_configs()
    all_results = {}
    for label, cfg in configs:
        runs_F, runs_time = run_config(problem, cfg, N_SEEDS, N_GEN)
        all_results[label] = (cfg, runs_F, runs_time)
        mean_archive = np.mean([len(f) for f in runs_F])
        print(f"done: {label:24s}  mean_time={np.mean(runs_time):.2f}s  mean_archive={mean_archive:.1f}")

    ref_point = (
        np.max(
            np.vstack([f for (_, runs_F, _) in all_results.values() for f in runs_F if len(f) > 0]),
            axis=0,
        )
        * 1.1
    )

    rows = []
    for label, (cfg, runs_F, runs_time) in all_results.items():
        for run_idx, F in enumerate(runs_F):
            if len(F) == 0:
                continue
            ind = all_indicators(F, ref_point)
            row_cfg = {k: (v if v is not None else "default") for k, v in cfg.items()}
            ind.update(config_label=label, run=run_idx, time_s=runs_time[run_idx], **row_cfg)
            rows.append(ind)
    df = pd.DataFrame(rows)

    out_dir = base / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print("\nHypervolume mean +- std per config, sorted best-first:")
    print(summary.round(4).to_string())

    baseline_hv = summary.loc["baseline", "mean"]
    print(f"\nbaseline mean HV over {N_SEEDS} seeds = {baseline_hv:.4f}")

    best_per_param = {}
    for param in SWEEP:
        sub = summary[summary.index.str.startswith(f"{param}=")]
        if len(sub) == 0:
            continue
        best_label = sub["mean"].idxmax()
        best_val = parse_value(best_label.split("=", 1)[1])
        best_per_param[param] = best_val
        print(f"  best {param}: {best_val} (mean HV {sub.loc[best_label, 'mean']:.4f} vs baseline {baseline_hv:.4f})")

    combined_cfg = dict(DEFAULTS)
    combined_cfg.update(best_per_param)
    print(f"\ncombined-best config: {combined_cfg}")

    runs_F, runs_time = run_config(problem, combined_cfg, COMBINED_SEEDS, N_GEN)
    combined_hv = [all_indicators(F, ref_point)["hypervolume"] for F in runs_F if len(F) > 0]
    print(
        f"combined-best mean HV over {COMBINED_SEEDS} seeds = {np.mean(combined_hv):.4f} "
        f"+- {np.std(combined_hv):.4f}  (baseline {baseline_hv:.4f} over {N_SEEDS} seeds)"
    )

    combined_rows = []
    for run_idx, F in enumerate(runs_F):
        if len(F) == 0:
            continue
        ind = all_indicators(F, ref_point)
        ind.update(config_label="combined-best", run=run_idx, time_s=runs_time[run_idx], **combined_cfg)
        combined_rows.append(ind)
    pd.concat([df, pd.DataFrame(combined_rows)], ignore_index=True).to_csv(
        out_dir / f"tune_qiea_{instance}.csv", index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instance", nargs="?", default="A-n32-k5")
    args = parser.parse_args()
    main(args.instance)
