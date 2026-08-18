"""
H/n_partitions scaling investigation (logs.txt section 15g-i, the last open
top-priority lever for QIEA's remaining gap to MOEA/D and RVEA).

Population size H is currently identical for QIEA/MOEA-D/RVEA (all three take
n_partitions=5 -> H=126 via Das-Dennis reference directions, matched for
fairness since section 8b) but NOT for NSGA-II/SPEA2, which use a separately
set pop_size (80-100) -- a known, previously-unaddressed mismatch. This script
varies n_partitions for ALL FIVE algorithms together at each grid point,
setting NSGA-II/SPEA2's pop_size = H too, so every comparison in the grid is
population-matched across all five algorithms -- closing that side mismatch
as well as testing the actual question this section is named for: does
giving every algorithm a larger or smaller population change QIEA's
hypervolume ratio to the best baseline, or does the whole field scale
together and the ratio stay flat?

Grid: n_partitions in {3,4,5,6,7,8} -> H in {35,70,126,210,330,495} (n_obj=5,
Das-Dennis simplex-lattice sizes). n_partitions=5/H=126 is the existing
default/baseline used everywhere else in this repo. n_gen=500 (the paper's
actual budget) -- per section 15b's lesson, population-size effects are
structural (interact with the plateau/restart mechanism) and must be
evaluated at the real generation budget, not an n_gen=80 pilot. 5 seeds per
config, matching every prior screen's convention in this file's siblings.

A timing probe (see logs.txt/commit history) showed cost scales roughly
linearly with H and is cheap even at H=495/n_gen=500 on the largest instance
tested here (route2_199) -- tens of seconds per run, not minutes.

One shared reference point per instance, across every H-level and every
algorithm run in this script (not recomputed per H-level) -- so hv_mean is
directly comparable across the whole grid, same convention as
tune_qiea_restart.py. Run per-instance (parallelizable, same convention as
tune_qiea_restart_confirm.py): `python tune_h_partitions_screen.py <instance>`
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from run_baselines import build_algorithms  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.util.ref_dirs import get_reference_directions  # noqa: E402

N_GEN = 500
N_SEEDS = 5
PARTITIONS_GRID = [3, 4, 5, 6, 7, 8]
BASELINE_NAMES = ["NSGA-II", "SPEA2", "MOEA/D", "RVEA"]
ALGO_NAMES = ["QIEA"] + BASELINE_NAMES


def h_for(n_partitions):
    return len(get_reference_directions("das-dennis", 5, n_partitions=n_partitions))


def run_instance(instance_name):
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{instance_name}.json"
    inst = CVRPInstance.from_file(inst_path)
    problem = CVRPProblem(inst)

    all_F = {}
    all_time = {}
    t0 = time.time()
    for np_ in PARTITIONS_GRID:
        H = h_for(np_)
        for seed in range(N_SEEDS):
            ts = time.time()
            qiea = QIEA(problem, decode="permutation", n_partitions=np_, seed=seed)
            all_F.setdefault((np_, "QIEA"), []).append(qiea.run(n_gen=N_GEN).F)
            all_time.setdefault((np_, "QIEA"), []).append(time.time() - ts)

            baselines = build_algorithms(problem.n_obj, pop_size=H, n_partitions=np_)
            for name, algo in baselines.items():
                ts = time.time()
                res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
                all_F.setdefault((np_, name), []).append(res.F)
                all_time.setdefault((np_, name), []).append(time.time() - ts)
        print(f"[{instance_name}] n_partitions={np_} (H={H}) done, {time.time() - t0:.1f}s elapsed", flush=True)

    ref_point = np.max(np.vstack([f for runs in all_F.values() for f in runs]), axis=0) * 1.1
    hv = HV(ref_point=ref_point)

    rows = []
    for np_ in PARTITIONS_GRID:
        H = h_for(np_)
        means = {}
        for name in ALGO_NAMES:
            hvs = [hv(F) for F in all_F[(np_, name)]]
            means[name] = np.mean(hvs)
            rows.append(
                dict(
                    instance=instance_name,
                    n_partitions=np_,
                    H=H,
                    algorithm=name,
                    hv_mean=means[name],
                    hv_std=np.std(hvs),
                    time_s_mean=np.mean(all_time[(np_, name)]),
                )
            )
        best_alg = max(BASELINE_NAMES, key=lambda n: means[n])
        ratio = means["QIEA"] / means[best_alg]
        for r in rows[-len(ALGO_NAMES):]:
            r["best_baseline"] = best_alg
            r["qiea_ratio_to_best"] = ratio

    df = pd.DataFrame(rows)
    print(
        df[df.algorithm == "QIEA"][["n_partitions", "H", "best_baseline", "qiea_ratio_to_best"]].to_string(
            index=False
        )
    )
    return df


if __name__ == "__main__":
    instance = sys.argv[1]
    df = run_instance(instance)
    out_dir = Path(__file__).resolve().parent.parent / "results"
    df.to_csv(out_dir / f"tune_h_partitions_screen_{instance}.csv", index=False)
