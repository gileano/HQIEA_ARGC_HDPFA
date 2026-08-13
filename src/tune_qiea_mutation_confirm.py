"""
Confirmatory run for the mutation_prob scaling formula found by
tune_qiea_mutation_scaling.py: expected mutated genes per individual
(mutation_prob * n_var) should stay roughly constant across instance sizes
instead of a fixed absolute mutation_prob getting more disruptive as n_var
grows -- the same class of fix as section 9's theta_min/theta_max scaling.

Candidate formula: mutation_prob = min(0.05, C / n_var), C=1.55 chosen so it
reduces EXACTLY to the old default 0.05 at n_var=31 (A-n32-k5, where 0.05 was
originally tuned) -- same clamping trick as _default_theta_bounds, motivated
by section 9e's finding that A-n32-k5 is sensitive to even ~1% drift in a
rotation-step-family parameter.

Also checks whether neighborhood_size (screened as separately helpful in
tune_qiea_large.py, at nb=50 for E-n101-k8 and nb=20 for route2_199) still
helps ON TOP of the corrected mutation_prob, or whether that screen result
was confounded by the same OFAT interaction section 8c hit -- i.e. compares
three configs per instance: default, mutation-formula-only, and
mutation-formula + the previously-screened neighborhood_size.
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
C = 1.55
N_SEEDS = 15
N_GEN = 80

# from tune_qiea_large.py's screen (best neighborhood_size per instance)
SCREENED_NB = {"A-n32-k5": 10, "E-n101-k8": 50, "route2_199": 20}


def mutation_formula(n_var):
    return min(0.05, C / n_var)


def run_config(problem, mutation_prob, neighborhood_size, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(
            problem,
            decode="permutation",
            seed=seed,
            n_partitions=FIXED_N_PARTITIONS,
            neighborhood_size=neighborhood_size,
            mutation_prob=mutation_prob,
        )
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)
    n_var = problem.n_var
    mp_formula = mutation_formula(n_var)
    nb_screened = SCREENED_NB[instance]

    configs = {
        "default": (0.05, 10),
        "mutation-formula-only": (mp_formula, 10),
        "mutation-formula+nb-screened": (mp_formula, nb_screened),
    }

    all_F, all_time = {}, {}
    for label, (mp, nb) in configs.items():
        runs_F, runs_time = run_config(problem, mp, nb, N_SEEDS, N_GEN)
        all_F[label] = runs_F
        all_time[label] = runs_time
        print(f"[{instance}, n_var={n_var}] done: {label:32s} mp={mp:.4f} nb={nb}  mean_time={np.mean(runs_time):.2f}s")

    ref_point = (
        np.max(np.vstack([f for runs in all_F.values() for f in runs if len(f) > 0]), axis=0) * 1.1
    )

    hv = {}
    rows = []
    for label, runs_F in all_F.items():
        vals = []
        for run_idx, F in enumerate(runs_F):
            if len(F) == 0:
                continue
            ind = all_indicators(F, ref_point)
            ind.update(config_label=label, run=run_idx, time_s=all_time[label][run_idx], instance=instance, n_var=n_var)
            rows.append(ind)
            vals.append(ind["hypervolume"])
        hv[label] = np.array(vals)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_mutation_confirm_{instance}.csv", index=False)

    print(f"\n[{instance}] mean HV: " + ", ".join(f"{k}={v.mean():.4e}" for k, v in hv.items()))

    n = min(len(hv["default"]), len(hv["mutation-formula-only"]))
    stat, p = stats.wilcoxon(hv["mutation-formula-only"][:n], hv["default"][:n])
    pct = 100.0 * (hv["mutation-formula-only"].mean() - hv["default"].mean()) / hv["default"].mean()
    print(f"[{instance}] mutation-formula-only vs default: {pct:+.1f}% HV, Wilcoxon p={p:.4f}")

    n2 = min(len(hv["mutation-formula-only"]), len(hv["mutation-formula+nb-screened"]))
    stat2, p2 = stats.wilcoxon(hv["mutation-formula+nb-screened"][:n2], hv["mutation-formula-only"][:n2])
    pct2 = (
        100.0
        * (hv["mutation-formula+nb-screened"].mean() - hv["mutation-formula-only"].mean())
        / hv["mutation-formula-only"].mean()
    )
    print(f"[{instance}] +nb-screened vs mutation-formula-only: {pct2:+.1f}% HV, Wilcoxon p={p2:.4f}")

    return {"instance": instance, "n_var": n_var, "mp_formula": mp_formula, "pct_vs_default": pct, "p_vs_default": p,
            "pct_nb_addon": pct2, "p_nb_addon": p2}


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(i, base, out_dir) for i in instances]
    print("\n=== SUMMARY ===")
    for s in summaries:
        sig = "significant" if s["p_vs_default"] < 0.05 else "not significant"
        nb_sig = "significant" if s["p_nb_addon"] < 0.05 else "not significant"
        print(
            f"{s['instance']:12s} n_var={s['n_var']:4d} mp_formula={s['mp_formula']:.4f}  "
            f"formula-only: {s['pct_vs_default']:+.1f}% (p={s['p_vs_default']:.4f}, {sig})  "
            f"+nb: {s['pct_nb_addon']:+.1f}% (p={s['p_nb_addon']:.4f}, {nb_sig})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["A-n32-k5", "E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
