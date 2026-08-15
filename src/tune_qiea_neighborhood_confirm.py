"""
Confirmatory run for tune_qiea_neighborhood_scaling.py's screen: takes the best
neighborhood_size found per instance there and re-tests it against the default (10)
with more seeds and a paired Wilcoxon test, mutation_prob/theta bounds left at their
current auto-scaled defaults (same as the screen).

BEST_NB below must be filled in from the screen's printed SUMMARY before running this.
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
DEFAULT_NB = 10
N_SEEDS = 15
N_GEN = 80

# Filled in from tune_qiea_neighborhood_scaling.py's screen output (5-seed screen):
# A-n32-k5 best=10 (=default, monotonically WORSE as nb grows there -- nothing to confirm);
# E-n101-k8 best=126=H (i.e. unrestricted mating, +50.4% HV, p=0.125 at 5 seeds);
# route2_199 best=20 (+54.6% HV, p=0.0625 at 5 seeds; note nb=126 is WORSE than nb=20 here,
# so E-n101-k8's and route2_199's optima are not even in the same direction relative to H,
# let alone tied to n_var -- route2_199 has the larger n_var but wants a much SMALLER
# neighborhood boost than E-n101-k8).
BEST_NB = {"A-n32-k5": None, "E-n101-k8": 126, "route2_199": 20}


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
    best_nb = BEST_NB[instance]

    configs = {"default": DEFAULT_NB}
    if best_nb is not None and best_nb != DEFAULT_NB:
        configs["screened-best"] = best_nb

    all_F, all_time = {}, {}
    for label, nb in configs.items():
        runs_F, runs_time = run_config(problem, nb, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:16s} nb={nb}  mean_time={np.mean(runs_time):.2f}s")

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
    df.to_csv(out_dir / f"tune_qiea_neighborhood_confirm_{instance}.csv", index=False)

    print(f"\n[{instance}] mean HV: " + ", ".join(f"{k}={v.mean():.4e}" for k, v in hv.items()))

    if "screened-best" not in hv:
        print(f"[{instance}] screen found default already best -- nothing to confirm")
        return {"instance": instance, "n_var": n_var, "best_nb": best_nb, "pct": 0.0, "p": 1.0}

    n = min(len(hv["default"]), len(hv["screened-best"]))
    stat, p = stats.wilcoxon(hv["screened-best"][:n], hv["default"][:n])
    pct = 100.0 * (hv["screened-best"].mean() - hv["default"].mean()) / hv["default"].mean()
    print(f"[{instance}] nb={best_nb} vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")
    return {"instance": instance, "n_var": n_var, "best_nb": best_nb, "pct": pct, "p": p}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY ===")
    for s in summaries:
        sig = "significant" if s["p"] < 0.05 else "not significant"
        print(f"{s['instance']:12s} n_var={s['n_var']:4d} best_nb={s['best_nb']}  {s['pct']:+.1f}% (p={s['p']:.4f}, {sig})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
