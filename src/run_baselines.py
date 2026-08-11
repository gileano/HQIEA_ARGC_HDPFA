"""
Run the classical multi-objective baselines (NSGA-II, SPEA2, MOEA/D, RVEA) on a
Sibiu CVRP instance, using the same permutation representation as the QIEA so
results are directly comparable. All four come straight from pymoo; only the
sampling/crossover/mutation operators are swapped to permutation-valid ones.
"""
import sys
import time
from pathlib import Path

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402

PERM_OPS = dict(
    sampling=PermutationRandomSampling(),
    crossover=OrderCrossover(),
    mutation=InversionMutation(),
)


def build_algorithms(n_obj, pop_size=100, n_partitions=5):
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    return {
        "NSGA-II": NSGA2(pop_size=pop_size, eliminate_duplicates=True, **PERM_OPS),
        "SPEA2": SPEA2(pop_size=pop_size, eliminate_duplicates=True, **PERM_OPS),
        "MOEA/D": MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9, **PERM_OPS),
        "RVEA": RVEA(ref_dirs=ref_dirs, eliminate_duplicates=True, **PERM_OPS),
    }


def run_all(instance_path, n_gen=100, seed=1, pop_size=100):
    inst = CVRPInstance.from_file(instance_path)
    problem = CVRPProblem(inst)
    algorithms = build_algorithms(problem.n_obj, pop_size=pop_size)

    results = {}
    for name, algo in algorithms.items():
        t0 = time.time()
        res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
        elapsed = time.time() - t0
        results[name] = {"F": res.F, "X": res.X, "time_s": elapsed}
        print(f"{name:10s} n_sol={len(res.F):4d}  t={elapsed:6.1f}s")
    return inst, results


if __name__ == "__main__":
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "route2_199.json"
    run_all(inst_path, n_gen=50, pop_size=80)
