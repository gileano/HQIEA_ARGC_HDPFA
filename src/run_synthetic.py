"""
Synthetic-benchmark harness: run the QIEA (continuous decode mode) and pymoo's
NSGA-II side by side on the standard ZDT/DTLZ/WFG suites, report hypervolume and
IGD against each problem's known Pareto front. This is the reviewer-expected
sanity layer before trusting results on the real Sibiu/CVRPLIB instances.
"""
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiea import QIEA  # noqa: E402

PROBLEMS = [
    ("zdt1", 2),
    ("zdt2", 2),
    ("zdt3", 2),
    ("dtlz1", 3),
    ("dtlz2", 3),
    ("wfg1", 3),
]


def run_nsga2(problem, n_gen, seed):
    algo = NSGA2(pop_size=100)
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    return res.F


def run_qiea(problem, n_gen, seed):
    algo = QIEA(problem, decode="continuous", seed=seed)
    res = algo.run(n_gen=n_gen)
    return res.F


def main(n_gen=100, seed=1):
    for name, n_obj in PROBLEMS:
        if name.startswith("wfg"):
            problem = get_problem(name, n_var=10, n_obj=n_obj)
        elif name.startswith("dtlz"):
            problem = get_problem(name, n_obj=n_obj)
        else:
            problem = get_problem(name)

        try:
            pf = problem.pareto_front()
        except Exception:
            pf = None

        t0 = time.time()
        F_nsga2 = run_nsga2(problem, n_gen, seed)
        t_nsga2 = time.time() - t0

        t0 = time.time()
        F_qiea = run_qiea(problem, n_gen, seed)
        t_qiea = time.time() - t0

        ref_point = np.max(np.vstack([F_nsga2, F_qiea]), axis=0) * 1.1
        hv = HV(ref_point=ref_point)
        hv_nsga2, hv_qiea = hv(F_nsga2), hv(F_qiea)

        igd_nsga2 = igd_qiea = float("nan")
        if pf is not None:
            igd = IGD(pf)
            igd_nsga2, igd_qiea = igd(F_nsga2), igd(F_qiea)

        print(
            f"{name:6s} obj={n_obj}  "
            f"NSGA2: HV={hv_nsga2:.4g} IGD={igd_nsga2:.4g} n={len(F_nsga2)} t={t_nsga2:.1f}s  |  "
            f"QIEA: HV={hv_qiea:.4g} IGD={igd_qiea:.4g} n={len(F_qiea)} t={t_qiea:.1f}s"
        )


if __name__ == "__main__":
    main()
