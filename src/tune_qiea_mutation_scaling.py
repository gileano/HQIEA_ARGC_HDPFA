"""
Follow-up to tune_qiea_large.py's OFAT screen, which found mutation_prob=0.01 and a
much larger neighborhood_size both looking dramatically better than baseline on
E-n101-k8/route2_199 (+59%/+109% HV) -- but combining them scored WORSE than
mutation_prob alone did in the screen, the same OFAT-interaction pattern section 8c
hit combining theta_min/theta_max/mutation_prob. This isolates mutation_prob from
neighborhood_size (held at the default 10) and tests a mechanistic scaling hypothesis:
quantum_mutation flips each gene independently with probability mutation_prob, so the
expected number of mutated genes per individual is mutation_prob * n_var -- a fixed
absolute mutation_prob is therefore an increasingly disruptive number of flips as
n_var grows, the same class of problem as section 9's fixed-absolute-theta-step
finding. Sweeps mutation_prob expressed as a target expected-mutations-per-individual
k = mutation_prob * n_var, i.e. mutation_prob = k / n_var, across all three instances
(A-n32-k5 included as the regression check -- it must NOT get worse there, since its
existing default was tuned on exactly that instance).
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
NEIGHBORHOOD_SIZE = 10  # held at default -- isolate mutation_prob only

K_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.9]  # 9.9 ~= default 0.05 at n_var=198 (route2_199)

N_SEEDS = 5
N_GEN = 80


def run_config(problem, mutation_prob, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            neighborhood_size=NEIGHBORHOOD_SIZE,
            mutation_prob=mutation_prob,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    default_mp = 0.05
    configs = [("default(0.05)", default_mp)]
    for k in K_GRID:
        mp = k / n_var
        configs.append((f"k={k}(mp={mp:.4f})", mp))

    all_F, all_time = {}, {}
    for label, mp in configs:
        runs_F, runs_time = run_config(problem, mp, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:22s} mean_time={np.mean(runs_time):.2f}s")

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
    df.to_csv(out_dir / f"tune_qiea_mutation_scaling_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"\n[{instance}, n_var={n_var}] Hypervolume mean +- std, sorted best-first:")
    print(summary.round(4).to_string())

    default_hv = df[df.config_label == "default(0.05)"]["hypervolume"].to_numpy()
    best_label = summary.index[0]
    best_hv = df[df.config_label == best_label]["hypervolume"].to_numpy()
    n = min(len(default_hv), len(best_hv))
    if best_label != "default(0.05)" and n >= 2:
        stat, p = stats.wilcoxon(best_hv[:n], default_hv[:n])
        pct = 100.0 * (best_hv.mean() - default_hv.mean()) / default_hv.mean()
        print(f"[{instance}] best={best_label} vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
    else:
        print(f"[{instance}] default(0.05) is already best")


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    for instance in instances:
        sweep_instance(instance, base, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
