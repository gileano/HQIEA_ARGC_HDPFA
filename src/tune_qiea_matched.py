"""
OFAT hyperparameter sweep for QIEA, holding population size fixed at parity
with the baselines it is compared against.

The first pass (tune_qiea.py) found n_partitions=8 (H=495) looked best, but
that is a false win: run_baselines.build_algorithms() uses n_partitions=5
(H=126) for BOTH MOEA/D and RVEA's reference directions, matching QIEA's own
default exactly. Raising QIEA's n_partitions alone would just buy it more
function evaluations per generation than the two baselines it currently loses
to, not a smarter search. So here n_partitions is held fixed at 5 (parity with
MOEA/D and RVEA in run_experiment.py) and only neighborhood_size, mutation_prob,
theta_max, and theta_min are retuned around that fixed, fair population.
theta_min's range is extended past the first sweep's 0.1 ceiling to check
whether that boundary result was a real optimum or just the edge of what was
tested.
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

FIXED_N_PARTITIONS = 5  # parity with MOEA/D + RVEA's ref_dirs in run_baselines.build_algorithms

DEFAULTS = dict(neighborhood_size=10, theta_max=0.35, theta_min=0.02, mutation_prob=0.05, n_partitions=FIXED_N_PARTITIONS)

SWEEP = {
    "neighborhood_size": [5, 10, 15, 20, 30, 50],
    "mutation_prob": [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
    "theta_max": [0.15, 0.25, 0.35, 0.5, 0.7, 0.9],
    "theta_min": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
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
    df.to_csv(out_dir / f"tune_qiea_matched_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print("\nHypervolume mean +- std per config, sorted best-first:")
    print(summary.round(4).to_string())

    baseline_hv = summary.loc["baseline", "mean"]
    print(f"\nbaseline (n_partitions=5, matched to MOEA/D+RVEA) mean HV over {N_SEEDS} seeds = {baseline_hv:.4f}")

    best_per_param = {}
    for param in SWEEP:
        sub = summary[summary.index.str.startswith(f"{param}=")]
        if len(sub) == 0:
            continue
        best_label = sub["mean"].idxmax()
        best_val = parse_value(best_label.split("=", 1)[1])
        beats_baseline = sub.loc[best_label, "mean"] > baseline_hv
        best_per_param[param] = best_val if beats_baseline else DEFAULTS[param]
        flag = "" if beats_baseline else "  (does NOT beat baseline -> keeping default)"
        print(f"  best {param}: {best_val} (mean HV {sub.loc[best_label, 'mean']:.4f} vs baseline {baseline_hv:.4f}){flag}")

    combined_cfg = dict(DEFAULTS)
    combined_cfg.update(best_per_param)
    print(f"\ncombined-best (matched-budget) config: {combined_cfg}")

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
        ind.update(config_label="combined-best-matched", run=run_idx, time_s=runs_time[run_idx], **combined_cfg)
        combined_rows.append(ind)
    pd.concat([df, pd.DataFrame(combined_rows)], ignore_index=True).to_csv(
        out_dir / f"tune_qiea_matched_{instance}.csv", index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instance", nargs="?", default="A-n32-k5")
    args = parser.parse_args()
    main(args.instance)
