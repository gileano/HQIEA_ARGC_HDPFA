"""
Diagnostic (not a tuning script): before sweeping diversity_window/diversity_stagnation_tol/
the boost multipliers, find out whether the diversity-stagnation escape in
QIEA.rotation_angle even fires under current defaults, and if so when/how often --
tuning parameters that never trigger would be pointless. Monkeypatches
QIEA.rotation_angle to record (gen, boosted, diversity) without changing qiea.py or the
algorithm's actual behavior at all (the wrapper calls the real method and just observes
its return value).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

N_SEEDS = 5
N_GEN = 80
FIXED_N_PARTITIONS = 5


def run_instrumented(problem, seed):
    algo = QIEA(problem, decode="permutation", seed=seed, n_partitions=FIXED_N_PARTITIONS)
    log = []
    orig = algo.rotation_angle

    def wrapped(gen, max_gen):
        step, boosted = orig(gen, max_gen)
        div = algo.history_diversity[-1] if algo.history_diversity else None
        log.append((gen, boosted, step, div))
        return step, boosted

    algo.rotation_angle = wrapped
    algo.run(n_gen=N_GEN)
    return log


def diagnose_instance(instance, base):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    boost_counts = []
    first_boost_gens = []
    for seed in range(N_SEEDS):
        log = run_instrumented(problem, seed)
        boosted_gens = [g for g, b, s, d in log if b]
        boost_counts.append(len(boosted_gens))
        first_boost_gens.append(boosted_gens[0] if boosted_gens else None)

    print(f"[{instance}, n_var={n_var}] boosted generations out of {N_GEN}, per seed: {boost_counts}")
    print(f"[{instance}, n_var={n_var}] first boost gen, per seed: {first_boost_gens}")
    print(f"[{instance}, n_var={n_var}] mean boosted fraction: {np.mean(boost_counts) / N_GEN:.2%}")


def main(instances):
    base = Path(__file__).resolve().parent.parent
    for instance in instances:
        diagnose_instance(instance, base)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
