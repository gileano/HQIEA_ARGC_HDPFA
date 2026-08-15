"""
Investigates the diversity-stagnation escape in QIEA.rotation_angle -- the "ARGC"
(Adaptive Rotation-gate Control) mechanism the algorithm/paper is named after.

Motivating diagnostic (diag_qiea_stagnation.py / diag_qiea_diversity_trace.py, run
before this script): at current defaults the stagnation boost almost NEVER fires --
0% of generations on A-n32-k5/E-n101-k8, ~1% on route2_199, across 80-generation runs.
diag_qiea_diversity_trace.py showed why: the windowed diversity range that
rotation_angle compares against diversity_stagnation_tol=1e-3 has a floor around
0.0017-0.003 across all three instances (never gets lower within 80 generations) --
i.e. the threshold is set ~2-3x too strict relative to where the metric actually
lives, so the entire stagnation-escape branch has been dead code in every result
reported in sections 6-10 of the implementation log. This screens
diversity_stagnation_tol across a grid spanning "never fires" (current default) up
through "fires often late-run" to test whether actually letting the mechanism engage
helps or hurts -- diversity_window and the two boost multipliers (now exposed as
constructor params, previously hardcoded 3.0/5.0) are held at their current defaults
in this first screen; mutation_prob/theta bounds left at their auto-scaled defaults.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_indicators  # noqa: E402
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

FIXED_N_PARTITIONS = 5
DEFAULT_TOL = 1e-3

# multiples of the current default, spanning "never fires" -> "fires often late-run"
TOL_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]

N_SEEDS = 5
N_GEN = 80


def run_config(problem, tol, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            diversity_stagnation_tol=tol,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    configs = [(f"tol={DEFAULT_TOL * m:.4f}" + ("(default)" if m == 1.0 else ""), DEFAULT_TOL * m) for m in TOL_GRID]

    all_F, all_time = {}, {}
    for label, tol in configs:
        runs_F, runs_time = run_config(problem, tol, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:20s} mean_time={np.mean(runs_time):.2f}s")

    ref_point = (
        np.max(np.vstack([f for runs in all_F.values() for f in runs if len(f) > 0]), axis=0) * 1.1
    )

    rows = []
    for label, runs_F in all_F.items():
        for run_idx, F in enumerate(runs_F):
            if len(F) == 0:
                continue
            ind = all_indicators(F, ref_point)
            ind.update(config_label=label, run=run_idx, time_s=all_time[label][run_idx], instance=instance, n_var=n_var)
            rows.append(ind)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_stagnation_scaling_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"\n[{instance}, n_var={n_var}] Hypervolume mean +- std, sorted best-first:")
    print(summary.round(4).to_string())

    default_label = f"tol={DEFAULT_TOL:.4f}(default)"
    default_hv = df[df.config_label == default_label]["hypervolume"].to_numpy()
    best_label = summary.index[0]
    best_hv = df[df.config_label == best_label]["hypervolume"].to_numpy()
    n = min(len(default_hv), len(best_hv))
    if best_label != default_label and n >= 2:
        stat, p = stats.wilcoxon(best_hv[:n], default_hv[:n])
        pct = 100.0 * (best_hv.mean() - default_hv.mean()) / default_hv.mean()
        print(f"[{instance}] best={best_label} vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
        best_tol = float(best_label.split("=")[1].split("(")[0])
    else:
        print(f"[{instance}] default tol={DEFAULT_TOL} is already best")
        best_tol = DEFAULT_TOL

    return {"instance": instance, "n_var": n_var, "best_tol": best_tol}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY (screen best diversity_stagnation_tol per instance) ===")
    for s in summaries:
        print(f"{s['instance']:12s} n_var={s['n_var']:4d}  best_tol={s['best_tol']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
