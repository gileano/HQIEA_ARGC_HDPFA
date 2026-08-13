"""
Tests whether QIEA's rotation-gate step size (theta_max/theta_min) needs to scale
with instance size (n_var) rather than stay fixed in absolute radians.

Motivation (see paper1.txt section 8f): tuning theta_min=0.1 on A-n32-k5 (n_var=31)
gave +15% hypervolume there but did NOT generalize to E-n101-k8 (n_var=100) or
route2_199 (n_var=198), where QIEA remained the weakest of the five algorithms.

Mechanism under test: permutation decode is tour = argsort(theta), over theta in
[0, pi/2]. The average gap between adjacent sorted theta values is ~ (pi/2)/n_var.
A rotation step that is a fixed absolute number of radians therefore represents a
shrinking or growing multiple of that gap as n_var changes -- e.g. theta_min=0.1 is
~2x the average gap at n_var=31, but ~13x the gap at n_var=198, i.e. a much more
disruptive relative move late in the run when the schedule should be exploiting,
not still reshuffling large chunks of the tour.

This script sweeps theta_min/theta_max expressed as multiples of that gap
(gap = pi/2 / n_var) across three instance sizes, and compares against the current
fixed-absolute defaults, to see whether a single gap-relative multiplier pair
generalizes across sizes where a single absolute pair does not.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_indicators  # noqa: E402
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402

FIXED_N_PARTITIONS = 5  # parity with MOEA/D + RVEA's ref_dirs in run_baselines.build_algorithms
FIXED_NEIGHBORHOOD = 10
FIXED_MUTATION_PROB = 0.05

INSTANCES = ["A-n32-k5", "E-n101-k8", "route2_199"]

# absolute reference configs (what's shipped today / what shipped before the 2026-08-12 tune)
ABSOLUTE_CONFIGS = {
    "absolute-current-default": dict(theta_min=0.1, theta_max=0.35),
    "absolute-pre-tune-default": dict(theta_min=0.02, theta_max=0.35),
}

# multiples of gap = (pi/2)/n_var. theta_min=0.1, theta_max=0.35 at n_var=31 correspond
# to ~2x and ~7x gap respectively, so the grid is centered there.
M_MIN = [0.5, 1, 2, 4, 8]
M_MAX = [2, 4, 7, 14, 28]

N_SEEDS_SCREEN = 3
N_SEEDS_CONFIRM = 10
N_GEN = 80


def run_config(problem, config, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            neighborhood_size=FIXED_NEIGHBORHOOD,
            mutation_prob=FIXED_MUTATION_PROB,
            **config,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def build_configs(n_var):
    gap = (np.pi / 2) / n_var
    configs = dict(ABSOLUTE_CONFIGS)
    for m_min in M_MIN:
        for m_max in M_MAX:
            theta_min = m_min * gap
            theta_max = m_max * gap
            if theta_min >= theta_max:
                continue
            configs[f"scaled-mmin={m_min}-mmax={m_max}"] = dict(theta_min=theta_min, theta_max=theta_max)
    return configs


def screen_instance(instance):
    base = Path(__file__).resolve().parent.parent
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = len(inst.customers)
    configs = build_configs(n_var)

    all_results = {}
    for label, cfg in configs.items():
        runs_F, runs_time = run_config(problem, cfg, N_SEEDS_SCREEN, N_GEN)
        all_results[label] = (cfg, runs_F, runs_time)

    ref_point = (
        np.max(
            np.vstack([f for (_, runs_F, _) in all_results.values() for f in runs_F if len(f) > 0]),
            axis=0,
        )
        * 1.1
    )

    rows = []
    for label, (cfg, runs_F, runs_time) in all_results.items():
        for run_idx, F in enumerate(runs_F):
            if len(F) == 0:
                continue
            ind = all_indicators(F, ref_point)
            ind.update(config_label=label, run=run_idx, time_s=runs_time[run_idx], instance=instance, n_var=n_var, **cfg)
            rows.append(ind)
    df = pd.DataFrame(rows)

    out_dir = base / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_scaling_{instance}.csv", index=False)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"\n=== {instance} (n_var={n_var}, gap={np.pi / 2 / n_var:.5f}) ===")
    print(summary.round(4).to_string())
    return instance, n_var, summary, ref_point, problem


def confirm_best(instance, problem, ref_point, label, cfg, n_seeds):
    runs_F, runs_time = run_config(problem, cfg, n_seeds, N_GEN)
    hv = [all_indicators(F, ref_point)["hypervolume"] for F in runs_F if len(F) > 0]
    print(
        f"  confirm {instance:12s} {label:32s} cfg={cfg}  mean HV over {n_seeds} seeds = "
        f"{np.mean(hv):.4f} +- {np.std(hv):.4f}"
    )
    return np.mean(hv), np.std(hv)


def main():
    screened = [screen_instance(inst) for inst in INSTANCES]

    print("\n\n================ cross-instance summary (screening, 3 seeds) ================")
    best_per_instance = {}
    for instance, n_var, summary, ref_point, problem in screened:
        best_label = summary["mean"].idxmax()
        best_per_instance[instance] = (best_label, summary.loc[best_label, "mean"])
        abs_best = summary.loc[[l for l in summary.index if l.startswith("absolute")]]["mean"].idxmax()
        print(
            f"{instance:12s} n_var={n_var:4d}  best overall: {best_label:32s} (HV={summary.loc[best_label, 'mean']:.4f})"
            f"  |  best absolute: {abs_best:28s} (HV={summary.loc[abs_best, 'mean']:.4f})"
        )

    print("\n================ confirming winners with more seeds ================")
    for instance, n_var, summary, ref_point, problem in screened:
        best_label, _ = best_per_instance[instance]
        cfgs_by_label = {}
        gap = (np.pi / 2) / n_var
        for m_min in M_MIN:
            for m_max in M_MAX:
                cfgs_by_label[f"scaled-mmin={m_min}-mmax={m_max}"] = dict(theta_min=m_min * gap, theta_max=m_max * gap)
        cfgs_by_label.update(ABSOLUTE_CONFIGS)

        confirm_best(instance, problem, ref_point, best_label, cfgs_by_label[best_label], N_SEEDS_CONFIRM)
        confirm_best(instance, problem, ref_point, "absolute-current-default", ABSOLUTE_CONFIGS["absolute-current-default"], N_SEEDS_CONFIRM)


if __name__ == "__main__":
    main()
