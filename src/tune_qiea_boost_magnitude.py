"""
Follow-up to tune_qiea_stagnation_scaling.py/tune_qiea_stagnation_confirm.py (section 12
of the implementation log), which found hypervolume decreasing monotonically as
diversity_stagnation_tol is raised to let the stagnation-escape boost fire more often --
at the boost's current fixed magnitude (3x rotation step, 5x mutation_prob). That
screen only varied how OFTEN the mechanism triggers, never HOW STRONG the response is
when it does, so it cannot distinguish two different explanations:
  (1) triggering the escape at all is harmful (the detector fires on noise, not real
      stagnation, so any override of the decaying schedule hurts), or
  (2) triggering is fine/beneficial, but a 3x/5x boost is simply too disruptive a
      response whenever it happens.

This fixes diversity_stagnation_tol=0.003 (confirmed in the prior screen to cause real,
consistent degradation across all three instances at the default 3x/5x multipliers --
i.e. a tol that reliably makes the mechanism engage) and sweeps
(rotation_boost_multiplier, mutation_boost_multiplier) jointly from a no-op (1.0, 1.0)
up to the current default (3.0, 5.0). (1.0, 1.0) is also a sanity check: it should
behave identically to the boost never firing at all (min(theta_max, base*1.0) == base,
mutation_prob*1.0 == mutation_prob), regardless of tol -- so it doubles as a check that
the harness reproduces the known no-boost baseline hypervolume from
tune_qiea_stagnation_scaling.py's tol=0.001(default) row.
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
FIXED_TOL = 0.003

# (rotation_boost_multiplier, mutation_boost_multiplier), from no-op up to the current default
MULT_GRID = [
    ("noop(1.0,1.0)", 1.0, 1.0),
    ("mild(1.1,1.2)", 1.1, 1.2),
    ("moderate(1.3,1.5)", 1.3, 1.5),
    ("strong(1.5,2.0)", 1.5, 2.0),
    ("stronger(2.0,3.0)", 2.0, 3.0),
    ("default(3.0,5.0)", 3.0, 5.0),
]

N_SEEDS = 5
N_GEN = 80


def run_config(problem, rot_mult, mut_mult, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            diversity_stagnation_tol=FIXED_TOL,
            rotation_boost_multiplier=rot_mult,
            mutation_boost_multiplier=mut_mult,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var

    all_F, all_time = {}, {}
    for label, rot_mult, mut_mult in MULT_GRID:
        runs_F, runs_time = run_config(problem, rot_mult, mut_mult, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:20s} mean_time={np.mean(runs_time):.2f}s")

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
    df.to_csv(out_dir / f"tune_qiea_boost_magnitude_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"])
    order = [label for label, _, _ in MULT_GRID]
    summary = summary.reindex(order)
    print(f"\n[{instance}, n_var={n_var}] Hypervolume mean +- std, in multiplier-grid order (noop -> default):")
    print(summary.round(4).to_string())

    noop_hv = df[df.config_label == "noop(1.0,1.0)"]["hypervolume"].to_numpy()
    default_hv = df[df.config_label == "default(3.0,5.0)"]["hypervolume"].to_numpy()
    n = min(len(noop_hv), len(default_hv))
    stat, p = stats.wilcoxon(default_hv[:n], noop_hv[:n])
    pct = 100.0 * (default_hv.mean() - noop_hv.mean()) / noop_hv.mean()
    print(f"[{instance}] default(3.0,5.0) vs noop(1.0,1.0): {pct:+.1f}% HV, Wilcoxon p={p:.4f}")

    best_label = summary["mean"].astype(float).idxmax()
    return {"instance": instance, "n_var": n_var, "best_label": best_label, "noop_hv": noop_hv.mean(), "default_hv": default_hv.mean()}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY (best multiplier config per instance, at tol=0.003) ===")
    for s in summaries:
        print(f"{s['instance']:12s} n_var={s['n_var']:4d}  best={s['best_label']:20s} "
              f"noop_hv={s['noop_hv']:.4e}  default_hv={s['default_hv']:.4e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
