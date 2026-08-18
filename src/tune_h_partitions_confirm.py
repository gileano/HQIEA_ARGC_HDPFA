"""
Confirmatory run for the H/n_partitions screen (tune_h_partitions_screen.py,
logs.txt section 15g-i). The screen was noisy and non-monotonic -- unlike the
restart-mechanism screen (section 15), NOT every larger H beat the current
default (n_partitions=5, H=126) on every instance. Ranking each grid point's
qiea_ratio_to_best per instance, n_partitions=7 (H=330) was the most robust
candidate: never worse than 2nd on any of the three screened instances
(A-n32-k5, E-n101-k8, route2_199), and outright best on two of three -- while
the single best-looking point on A-n32-k5 (H=495) was actually WORSE than the
current default on E-n101-k8, the same "great on one instance, does not
generalize" failure mode that kept neighborhood_size out of the shipped
defaults (section 11).

This script confirms ONE fixed candidate (n_partitions=7, H=330, applied to
all five algorithms together -- QIEA, MOEA/D, RVEA via n_partitions, NSGA-II/
SPEA2 via pop_size=H, same fairness treatment as the screen) against the
n_partitions=5/H=126 control, per-seed-paired, with more seeds (15) and a
proper Wilcoxon test. The paired quantity is QIEA's hypervolume ratio to the
best baseline AT THE SAME H -- not QIEA's raw hypervolume -- since population
size changes every algorithm's hypervolume, not just QIEA's; the question is
whether the RATIO improves, matching the screen's own metric.

Reference point per instance spans both configs (candidate + control, all
five algorithms, all seeds) so hv values are directly comparable across the
two conditions, same convention as tune_qiea_restart_confirm.py.

Run per-instance (parallelizable): `python tune_h_partitions_confirm.py <instance>`
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from run_baselines import build_algorithms  # noqa: E402
from metrics import wilcoxon_test  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.util.ref_dirs import get_reference_directions  # noqa: E402

N_GEN = 500
N_SEEDS = 15
CONTROL_NP = 5    # H=126, current shipped default
CANDIDATE_NP = 7  # H=330, most robust screen candidate
BASELINE_NAMES = ["NSGA-II", "SPEA2", "MOEA/D", "RVEA"]
ALGO_NAMES = ["QIEA"] + BASELINE_NAMES


def h_for(n_partitions):
    return len(get_reference_directions("das-dennis", 5, n_partitions=n_partitions))


def run_condition(problem, n_partitions, n_seeds):
    H = h_for(n_partitions)
    all_F = {name: [] for name in ALGO_NAMES}
    for seed in range(n_seeds):
        qiea = QIEA(problem, decode="permutation", n_partitions=n_partitions, seed=seed)
        all_F["QIEA"].append(qiea.run(n_gen=N_GEN).F)

        baselines = build_algorithms(problem.n_obj, pop_size=H, n_partitions=n_partitions)
        for name, algo in baselines.items():
            res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
            all_F[name].append(res.F)
    return all_F


def run_instance(instance_name):
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{instance_name}.json"
    inst = CVRPInstance.from_file(inst_path)
    problem = CVRPProblem(inst)

    t0 = time.time()
    F_control = run_condition(problem, CONTROL_NP, N_SEEDS)
    print(f"[{instance_name}] control (n_partitions={CONTROL_NP}) done, {time.time() - t0:.1f}s", flush=True)
    F_candidate = run_condition(problem, CANDIDATE_NP, N_SEEDS)
    elapsed = time.time() - t0
    print(f"[{instance_name}] candidate (n_partitions={CANDIDATE_NP}) done, {elapsed:.1f}s total", flush=True)

    ref_point = np.max(
        np.vstack([f for runs in list(F_control.values()) + list(F_candidate.values()) for f in runs]),
        axis=0,
    ) * 1.1
    hv = HV(ref_point=ref_point)

    def ratios(all_F):
        out = []
        for seed in range(N_SEEDS):
            hvs = {name: hv(all_F[name][seed]) for name in ALGO_NAMES}
            best = max(hvs[n] for n in BASELINE_NAMES)
            out.append(hvs["QIEA"] / best)
        return np.array(out)

    ratio_control = ratios(F_control)
    ratio_candidate = ratios(F_candidate)
    w = wilcoxon_test(ratio_candidate, ratio_control)
    pct = (ratio_candidate.mean() / ratio_control.mean() - 1) * 100

    print(
        f"[{instance_name}] n={N_SEEDS}  "
        f"ratio_control={ratio_control.mean():.4f}  ratio_candidate={ratio_candidate.mean():.4f}  "
        f"pct={pct:+.1f}%  wilcoxon p={w['p_value']:.4g}"
    )

    return pd.DataFrame(
        [
            dict(
                instance=instance_name,
                control_n_partitions=CONTROL_NP,
                candidate_n_partitions=CANDIDATE_NP,
                ratio_control_mean=ratio_control.mean(),
                ratio_control_std=ratio_control.std(),
                ratio_candidate_mean=ratio_candidate.mean(),
                ratio_candidate_std=ratio_candidate.std(),
                pct_change=pct,
                wilcoxon_stat=w["statistic"],
                wilcoxon_p=w["p_value"],
            )
        ]
    )


if __name__ == "__main__":
    instance = sys.argv[1]
    df = run_instance(instance)
    out_dir = Path(__file__).resolve().parent.parent / "results"
    df.to_csv(out_dir / f"tune_h_partitions_confirm_{instance}.csv", index=False)
