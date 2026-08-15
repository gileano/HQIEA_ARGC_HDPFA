"""
Confirmatory run for tune_qiea_boost_magnitude.py's screen. The screen's headline
result was not "smaller magnitude is proportionally better" (the middle of the grid
was noisy/non-monotonic on two of three instances) but that the MILDEST tested boost,
(rotation_boost_multiplier, mutation_boost_multiplier)=(1.1, 1.2), already
underperformed fully disabling the mechanism ((1.0, 1.0), a mathematical no-op
regardless of tol) on ALL THREE instances at tol=0.003 -- suggesting triggering the
escape at all tends to hurt, not just today's specific 3x/5x magnitude. Confirms
mild(1.1,1.2) vs noop(1.0,1.0) with more seeds.
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
FIXED_TOL = 0.003
N_SEEDS = 15
N_GEN = 80


def run_config(problem, rot_mult, mut_mult, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            diversity_stagnation_tol=FIXED_TOL,
            rotation_boost_multiplier=rot_mult,
            mutation_boost_multiplier=mut_mult,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    configs = {"noop(1.0,1.0)": (1.0, 1.0), "mild(1.1,1.2)": (1.1, 1.2)}

    all_F, all_time = {}, {}
    for label, (rot_mult, mut_mult) in configs.items():
        runs_F, runs_time = run_config(problem, rot_mult, mut_mult, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:16s} mean_time={np.mean(runs_time):.2f}s")

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
    df.to_csv(out_dir / f"tune_qiea_boost_magnitude_confirm_{instance}.csv", index=False)

    print(f"\n[{instance}] mean HV: " + ", ".join(f"{k}={v.mean():.4e}" for k, v in hv.items()))

    n = min(len(hv["noop(1.0,1.0)"]), len(hv["mild(1.1,1.2)"]))
    stat, p = stats.wilcoxon(hv["mild(1.1,1.2)"][:n], hv["noop(1.0,1.0)"][:n])
    pct = 100.0 * (hv["mild(1.1,1.2)"].mean() - hv["noop(1.0,1.0)"].mean()) / hv["noop(1.0,1.0)"].mean()
    print(f"[{instance}] mild(1.1,1.2) vs noop(1.0,1.0): {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
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
