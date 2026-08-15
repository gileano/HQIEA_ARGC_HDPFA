"""
Confirmatory run for tune_qiea_stagnation_scaling.py's screen, which found hypervolume
DECREASING monotonically as diversity_stagnation_tol increases (i.e. as the
stagnation-escape boost is allowed to trigger more often), on all three instances at 5
seeds -- the opposite of a lever to tune upward. Confirms the smallest deviation from
default (tol=0.002, 2x the default 0.001) with more seeds, since a small perturbation
is the case most likely to be noise; if even that is significantly worse, the
monotonic-harm finding is robust and not a large-perturbation artifact.
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
MILD_TOL = 2e-3
N_SEEDS = 15
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

    configs = {"default(0.001)": DEFAULT_TOL, "mild(0.002)": MILD_TOL}

    all_F, all_time = {}, {}
    for label, tol in configs.items():
        runs_F, runs_time = run_config(problem, tol, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:16s} tol={tol}  mean_time={np.mean(runs_time):.2f}s")

    ref_point = (
        np.max(np.vstack([f for runs in all_F.values() for f in runs if len(f) > 0]), axis=0) * 1.1
    )

    hv = {}
    rows = []
    for label, runs_F in all_F.items():
        vals = []
        for run_idx, F in enumerate(runs_F):
            if len(F) == 0:
                continue
            ind = all_indicators(F, ref_point)
            ind.update(config_label=label, run=run_idx, time_s=all_time[label][run_idx], instance=instance, n_var=n_var)
            rows.append(ind)
            vals.append(ind["hypervolume"])
        hv[label] = np.array(vals)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_stagnation_confirm_{instance}.csv", index=False)

    print(f"\n[{instance}] mean HV: " + ", ".join(f"{k}={v.mean():.4e}" for k, v in hv.items()))

    n = min(len(hv["default(0.001)"]), len(hv["mild(0.002)"]))
    stat, p = stats.wilcoxon(hv["mild(0.002)"][:n], hv["default(0.001)"][:n])
    pct = 100.0 * (hv["mild(0.002)"].mean() - hv["default(0.001)"].mean()) / hv["default(0.001)"].mean()
    print(f"[{instance}] mild(0.002) vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
    return {"instance": instance, "n_var": n_var, "pct": pct, "p": p}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY ===")
    for s in summaries:
        sig = "significant" if s["p"] < 0.05 else "not significant"
        print(f"{s['instance']:12s} n_var={s['n_var']:4d}  {s['pct']:+.1f}% (p={s['p']:.4f}, {sig})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
