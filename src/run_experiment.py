"""
Full comparison harness: QIEA vs NSGA-II/SPEA2/MOEA-D/RVEA on one instance,
across multiple seeds, with hypervolume/spacing/spread per run and Wilcoxon +
Friedman tests across algorithms. No true Pareto front exists for real CVRP
instances, so IGD/IGD+ are only reported for the synthetic suite (run_synthetic.py);
here quality is judged by hypervolume/spacing/spread against a shared reference
point, which is the standard protocol when the true front is unknown.

This is deliberately runnable at small (n_runs, n_gen) for a fast pilot, and at
paper-scale (30 runs, 500-1000 generations) for the final results -- the caller
picks the budget; this script does not hardcode "30 runs" because that scale
should be a conscious compute-time decision, not a silent default.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_indicators, friedman_test, wilcoxon_test  # noqa: E402
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from run_baselines import build_algorithms  # noqa: E402

from pymoo.optimize import minimize  # noqa: E402


def run_one(instance_path, n_gen, n_runs, pop_size=100, out_dir=None):
    inst = CVRPInstance.from_file(instance_path)
    problem = CVRPProblem(inst)
    algo_names = ["QIEA", "NSGA-II", "SPEA2", "MOEA/D", "RVEA"]

    all_F = {name: [] for name in algo_names}
    all_time = {name: [] for name in algo_names}

    for seed in range(n_runs):
        t0 = time.time()
        qiea = QIEA(problem, decode="permutation", seed=seed)
        res = qiea.run(n_gen=n_gen)
        all_F["QIEA"].append(res.F)
        all_time["QIEA"].append(time.time() - t0)

        baselines = build_algorithms(problem.n_obj, pop_size=pop_size)
        for name, algo in baselines.items():
            t0 = time.time()
            res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
            all_F[name].append(res.F)
            all_time[name].append(time.time() - t0)

        print(f"[{inst.name}] seed {seed+1}/{n_runs} done")

    ref_point = np.max(np.vstack([f for runs in all_F.values() for f in runs]), axis=0) * 1.1

    rows = []
    for name in algo_names:
        for run_idx, F in enumerate(all_F[name]):
            ind = all_indicators(F, ref_point)
            ind.update(algorithm=name, run=run_idx, instance=inst.name, time_s=all_time[name][run_idx])
            rows.append(ind)
    df = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{inst.name}_indicators.csv", index=False)
        np.savez(
            out_dir / f"{inst.name}_fronts.npz",
            **{f"{name}_run{i}": F for name in algo_names for i, F in enumerate(all_F[name])},
        )

    return df, ref_point


def statistical_summary(df):
    pivot = df.pivot(index="run", columns="algorithm", values="hypervolume")
    algo_names = list(pivot.columns)
    print("\nHypervolume mean +- std per algorithm:")
    print(pivot.mean().round(3).to_string(), "\n")
    print(pivot.std().round(3).rename("std").to_string())

    if len(pivot) >= 2 and len(algo_names) >= 2:
        friedman = friedman_test(pivot[algo_names].values)
        print(f"\nFriedman test across algorithms: chi2={friedman['statistic']:.3f} p={friedman['p_value']:.4g}")

        base = algo_names[0]
        print(f"\nWilcoxon signed-rank vs {base}:")
        for other in algo_names[1:]:
            try:
                w = wilcoxon_test(pivot[base].values, pivot[other].values)
                print(f"  {base} vs {other}: stat={w['statistic']:.3f} p={w['p_value']:.4g}")
            except ValueError as e:
                print(f"  {base} vs {other}: {e}")
    else:
        print("\n(need >=2 runs to compute Wilcoxon/Friedman)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=100)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    inst_path = base / "data" / "processed" / f"{args.instance}.json"
    out_dir = base / "results"

    df, ref_point = run_one(inst_path, args.n_gen, args.n_runs, args.pop_size, out_dir)
    statistical_summary(df)
