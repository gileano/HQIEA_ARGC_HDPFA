"""
Diagnostic (no tuning, no source changes): where does a QIEA run spend its wall-clock
time? Step 1 of the parallel-performance plan in submission2.txt section 7 -- the
share of time in fitness evaluation bounds any speedup from parallelizing it
(Amdahl), so this must be measured before building a parallel evaluator.

Two measurements per (instance, seed), each a full QIEA.run() at current defaults:
  1. Phase timing with low-overhead perf_counter wrappers (no cProfile), which gives
     the real shares:
       evaluate  -- CVRPProblem._evaluate (route split + 5 objectives), including the
                    evaluations done inside _do_restart
       archive   -- QIEA._update_archive (decode + np.unique + non-dominated pruning)
       other     -- everything else: per-subproblem Tchebycheff guide selection,
                    crossover/rotation/mutation, neighborhood replacement, diversity,
                    decode inside evaluate_population
  2. One cProfile run (first seed only) for a per-function breakdown. cProfile adds
     per-call overhead that inflates pure-Python code, so use it for ranking hot
     spots, not for the shares.

Run with BLAS pinned to one thread (the parallel work will run one process per core):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python src/diag_qiea_profile.py --n-gen 500 --n-seeds 3
Add --no-restart to time the same runs with the restart mechanism disabled.
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_INSTANCES = ["A-n32-k5", "route2_199", "route1_334"]


def _wrap(obj, name, acc, key):
    """Replace obj.name with a timed version that adds its elapsed time to acc[key]."""
    orig = getattr(obj, name)

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            acc[key] += time.perf_counter() - t0

    setattr(obj, name, timed)


def time_phases(inst, n_gen, seed, qiea_kwargs=None):
    problem = CVRPProblem(inst)
    algo = QIEA(problem, decode="permutation", seed=seed, **(qiea_kwargs or {}))
    acc = {"evaluate": 0.0, "archive": 0.0}
    counts = {"evals": 0}
    orig_eval = problem._evaluate

    def counted_eval(X, out, *a, **k):
        counts["evals"] += len(X)
        return orig_eval(X, out, *a, **k)

    problem._evaluate = counted_eval
    _wrap(problem, "_evaluate", acc, "evaluate")
    _wrap(algo, "_update_archive", acc, "archive")

    t0 = time.perf_counter()
    algo.run(n_gen)
    total = time.perf_counter() - t0
    other = total - acc["evaluate"] - acc["archive"]
    return {
        "total": total,
        "evaluate": acc["evaluate"],
        "archive": acc["archive"],
        "other": other,
        "evals": counts["evals"],
        "restarts": algo.n_restarts,
        "archive_size": len(algo.archive_F),
    }


def cprofile_run(inst, n_gen, seed, top=15, qiea_kwargs=None):
    problem = CVRPProblem(inst)
    algo = QIEA(problem, decode="permutation", seed=seed, **(qiea_kwargs or {}))
    prof = cProfile.Profile()
    prof.enable()
    algo.run(n_gen)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(top)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--no-cprofile", action="store_true")
    ap.add_argument("--no-restart", action="store_true",
                    help="restart_patience=None, to measure the restart mechanism's cost")
    args = ap.parse_args()
    kw = {"restart_patience": None} if args.no_restart else {}

    rows = []
    for name in args.instances:
        inst = CVRPInstance.from_file(DATA / f"{name}.json")
        n_var = len(inst.customers)
        runs = [time_phases(inst, args.n_gen, seed, kw) for seed in range(1, args.n_seeds + 1)]
        med = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
        rows.append((name, n_var, med, runs))
        print(f"\n=== {name} (n_var={n_var}, n_gen={args.n_gen}, {args.n_seeds} seeds) ===")
        for i, r in enumerate(runs, 1):
            print(
                f"  seed {i}: total {r['total']:7.2f}s  evaluate {r['evaluate']:7.2f}s  "
                f"archive {r['archive']:6.2f}s  other {r['other']:6.2f}s  "
                f"evals {r['evals']}  restarts {r['restarts']}  |archive| {r['archive_size']}"
            )
        if not args.no_cprofile:
            print(f"\n  cProfile (seed 1, sorted by own time -- ranking only, shares inflated):")
            print(cprofile_run(inst, args.n_gen, 1, qiea_kwargs=kw))

    print("\n=== Summary (medians over seeds) ===")
    print(f"{'instance':12s} {'n_var':>5s} {'total s':>8s} {'eval %':>7s} {'archive %':>9s} "
          f"{'other %':>8s} {'ms/eval':>8s} {'Amdahl max (eval only)':>23s}")
    for name, n_var, m, _ in rows:
        p = m["evaluate"] / m["total"]
        print(
            f"{name:12s} {n_var:5d} {m['total']:8.2f} {100 * p:6.1f}% {100 * m['archive'] / m['total']:8.1f}% "
            f"{100 * m['other'] / m['total']:7.1f}% {1000 * m['evaluate'] / m['evals']:8.3f} "
            f"{1 / (1 - p):22.1f}x"
        )


if __name__ == "__main__":
    main()
