"""
Environment sanity check: load a Sibiu instance, run a couple of NSGA-II
generations on a placeholder permutation-encoded CVRP-as-TSP objective pair
(distance, emissions) just to prove pymoo + numpy + the data pipeline work
together end to end. Not the real algorithm/encoding for the paper.
"""
import json
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "route2_199.json"


class TourObjectives(Problem):
    def __init__(self, dist, emit):
        super().__init__(n_var=dist.shape[0], n_obj=2, xl=0, xu=dist.shape[0] - 1, vtype=int)
        self.dist = dist
        self.emit = emit

    def _evaluate(self, X, out, *args, **kwargs):
        d = np.zeros(X.shape[0])
        e = np.zeros(X.shape[0])
        for i, tour in enumerate(X):
            nxt = np.roll(tour, -1)
            d[i] = self.dist[tour, nxt].sum()
            e[i] = self.emit[tour, nxt].sum()
        out["F"] = np.column_stack([d, e])


def main():
    inst = json.loads(DATA.read_text())
    dist = np.array(inst["distance_matrix_km"])
    emit = np.array(inst["emission_matrix_kgco2"])
    print(f"loaded {inst['name']}: n={inst['num_points']}")

    problem = TourObjectives(dist, emit)
    algorithm = NSGA2(
        pop_size=40,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", 20), seed=1, verbose=False)
    print(f"pareto front size: {len(res.F)}")
    print("sample objective pairs (distance km, emissions kgCO2):")
    for row in res.F[:5]:
        print(f"  {row[0]:.1f}, {row[1]:.1f}")


if __name__ == "__main__":
    main()
