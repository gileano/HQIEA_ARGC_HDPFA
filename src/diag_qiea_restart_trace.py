"""
Diagnostic (no tuning, no source changes): hypervolume-vs-generation for QIEA WITH and
WITHOUT the plateau-gated restart (logs.txt section 15), plus the four pymoo baselines,
over a full run. Produces the before/after plateau figure for the PDP 2027 paper.

Unlike diag_qiea_generation_trace.py (which re-implements the pre-restart loop and so
can only trace restart-free QIEA), this hooks QIEA._update_archive -- called once
after initialization and once per generation inside QIEA.run() -- so the real run()
is traced, restart included.

Reference point: fixed per (instance, seed), 1.1x the nadir of QIEA's initial random
population, shared by every algorithm and checkpoint in that run (same convention as
diag_qiea_generation_trace.py). HV values are comparable within one CSV only.

Output: results/trace_restart_<instance>_seed<k>.csv with columns
gen, QIEA, QIEA-norestart, NSGA-II, SPEA2, MOEA/D, RVEA.

  OMP_NUM_THREADS=1 python src/diag_qiea_restart_trace.py route2_199 --seed 1 --n-gen 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from run_baselines import build_algorithms  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

CHECKPOINT_EVERY = 10
BASE = Path(__file__).resolve().parent.parent


def trace_qiea(problem, n_gen, seed, hv, **kwargs):
    algo = QIEA(problem, decode="permutation", seed=seed, **kwargs)
    orig = algo._update_archive
    trace, calls = [], [0]

    def hooked(theta_pop, F):
        orig(theta_pop, F)
        gen = calls[0]  # call 0 = initial population, call g = after generation g
        calls[0] += 1
        if gen % CHECKPOINT_EVERY == 0 or gen == n_gen:
            trace.append((gen, float(hv(np.array(algo.archive_F)))))

    algo._update_archive = hooked
    algo.run(n_gen)
    return dict(trace)


def trace_baseline(problem, name, n_gen, seed, hv):
    algo = build_algorithms(problem.n_obj, pop_size=80)[name]
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False, save_history=True)
    return {h.n_gen: float(hv(h.opt.get("F"))) for h in res.history
            if h.n_gen % CHECKPOINT_EVERY == 0 or h.n_gen == n_gen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-gen", type=int, default=500)
    args = ap.parse_args()

    inst = CVRPInstance.from_file(BASE / "data" / "processed" / f"{args.instance}.json")
    problem = CVRPProblem(inst)
    probe = QIEA(problem, decode="permutation", seed=args.seed)
    _, F0 = probe.evaluate_population(probe.theta)
    hv = HV(ref_point=F0.max(axis=0) * 1.1)

    cols = {
        "QIEA": trace_qiea(problem, args.n_gen, args.seed, hv),
        "QIEA-norestart": trace_qiea(problem, args.n_gen, args.seed, hv, restart_patience=None),
    }
    for name in ["NSGA-II", "SPEA2", "MOEA/D", "RVEA"]:
        cols[name] = trace_baseline(problem, name, args.n_gen, args.seed, hv)

    df = pd.DataFrame(cols)
    df.index.name = "gen"
    out = BASE / "results" / f"trace_restart_{args.instance}_seed{args.seed}.csv"
    df.sort_index().to_csv(out)
    print(f"wrote {out}")
    print(df.sort_index().iloc[[0, len(df) // 4, len(df) // 2, -1]].to_string())


if __name__ == "__main__":
    main()
