"""
Follow-up to diag_qiea_stagnation.py's finding that the stagnation-escape boost almost
never fires at current defaults. Prints the raw diversity() trajectory and the
sliding-window range (max(recent)-min(recent), the exact quantity rotation_angle
compares against diversity_stagnation_tol=1e-3) to see what scale that quantity
actually operates at per instance -- needed to pick a sensible tol grid instead of
guessing blindly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

N_GEN = 80
FIXED_N_PARTITIONS = 5
WINDOW = 10


def trace_instance(instance, base, seed=0):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    algo = QIEA(problem, decode="permutation", seed=seed, n_partitions=FIXED_N_PARTITIONS)
    algo.run(n_gen=N_GEN)
    div = np.array(algo.history_diversity)

    ranges = [np.max(div[i - WINDOW : i]) - np.min(div[i - WINDOW : i]) for i in range(WINDOW, len(div) + 1)]
    ranges = np.array(ranges)

    print(f"[{instance}, n_var={n_var}] diversity: gen0={div[0]:.5f} gen10={div[min(10,len(div)-1)]:.5f} "
          f"gen40={div[min(40,len(div)-1)]:.5f} gen79={div[-1]:.5f}")
    print(f"[{instance}, n_var={n_var}] windowed range (what tol is compared against): "
          f"min={ranges.min():.6f} median={np.median(ranges):.6f} max={ranges.max():.6f}")
    print(f"[{instance}, n_var={n_var}] theta_min={algo.theta_min:.5f} theta_max={algo.theta_max:.5f} "
          f"mutation_prob={algo.mutation_prob:.5f}")


def main(instances):
    base = Path(__file__).resolve().parent.parent
    for instance in instances:
        trace_instance(instance, base)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
