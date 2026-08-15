"""
Follow-up to section 10's mutation_prob scaling fix, answering the open question left
in tune_qiea_mutation_confirm.py: with mutation_prob (and theta_min/theta_max) held at
their current auto-scaled defaults, does neighborhood_size have a clean size-based
optimum (tied to n_var, or to H via n_partitions), or is its earlier-observed effect
(+30% on route2_199, 0% on E-n101-k8, both from tune_qiea_mutation_confirm.py) instance-
specific?

Note H is IDENTICAL across all three CVRP instances tested here (n_partitions=5,
n_obj=5 -> H=126 always, per section 8b's fairness constraint that keeps QIEA's
population matched to MOEA/D/RVEA) -- so within this instance set, "tied to H" and
"tied to n_var" cannot be distinguished from each other; what CAN be checked is whether
the optimal neighborhood_size scales monotonically with n_var (A-n32-k5 n_var=31 <
E-n101-k8 n_var=100 < route2_199 n_var=198). The prior ad hoc screen (tune_qiea_large.py)
already contradicts a naive n_var-monotonic story -- it picked nb=50 for the
n_var=100 instance but only nb=20 for the n_var=198 instance -- but that screen had
mutation_prob wrong at the time, so this repeats it cleanly with mutation_prob fixed.

mutation_prob and theta_min/theta_max are left as None (QIEA's own auto-scaled
defaults) rather than pinned to specific values, since the goal is to characterize
neighborhood_size under the algorithm's actual shipped configuration, not in isolation
from it.
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

FIXED_N_PARTITIONS = 5  # matched to MOEA/D + RVEA, per section 8b
DEFAULT_NB = 10

NB_GRID = [5, 10, 15, 20, 30, 40, 60, 80, 126]  # H=126 at n_partitions=5/n_obj=5, so 126 = "mate with anyone"

N_SEEDS = 5
N_GEN = 80


def run_config(problem, neighborhood_size, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            neighborhood_size=neighborhood_size,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    configs = [(f"nb={nb}" + ("(default)" if nb == DEFAULT_NB else ""), nb) for nb in NB_GRID]

    all_F, all_time = {}, {}
    for label, nb in configs:
        runs_F, runs_time = run_config(problem, nb, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:16s} mean_time={np.mean(runs_time):.2f}s")

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
    df.to_csv(out_dir / f"tune_qiea_neighborhood_scaling_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"\n[{instance}, n_var={n_var}] Hypervolume mean +- std, sorted best-first:")
    print(summary.round(4).to_string())

    default_label = f"nb={DEFAULT_NB}(default)"
    default_hv = df[df.config_label == default_label]["hypervolume"].to_numpy()
    best_label = summary.index[0]
    best_hv = df[df.config_label == best_label]["hypervolume"].to_numpy()
    n = min(len(default_hv), len(best_hv))
    if best_label != default_label and n >= 2:
        stat, p = stats.wilcoxon(best_hv[:n], default_hv[:n])
        pct = 100.0 * (best_hv.mean() - default_hv.mean()) / default_hv.mean()
        print(f"[{instance}] best={best_label} vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
        best_nb = int(best_label.split("=")[1].split("(")[0])
    else:
        print(f"[{instance}] default nb={DEFAULT_NB} is already best")
        best_nb = DEFAULT_NB

    return {"instance": instance, "n_var": n_var, "best_nb": best_nb}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY (screen best nb per instance) ===")
    for s in summaries:
        print(f"{s['instance']:12s} n_var={s['n_var']:4d}  best_nb={s['best_nb']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
