"""
Confirmatory run for the plateau-gated diversity-reinjection mechanism
(restart_patience/restart_fraction, qiea.py). tune_qiea_restart.py and
tune_qiea_restart_finegrid.py both found EVERY tested config beats the
restart-disabled control on all three instances screened (A-n32-k5, E-n101-k8,
route2_199), but the per-instance-optimal patience did not agree (10, 5, 15
respectively) -- no clean size-based formula, same situation as neighborhood_size
in paper1.txt section 11. restart_patience=10/restart_fraction=0.5 was the most
robust single choice by average rank (top-3 on all three instances screened,
never an outlier) -- this script confirms that ONE fixed candidate vs control
with more seeds, and extends coverage to all 7 instances (only 3 were screened).

Run per-instance (parallelizable across instances, same convention as
run_experiment.py's full campaign): `python tune_qiea_restart_confirm.py <instance>`
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from metrics import wilcoxon_test  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402

N_GEN = 500
N_SEEDS = 20
CANDIDATE = dict(restart_patience=10, restart_fraction=0.5)


def run_instance(instance_name):
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{instance_name}.json"
    inst = CVRPInstance.from_file(inst_path)
    problem = CVRPProblem(inst)

    F_control, F_candidate = [], []
    t0 = time.time()
    for seed in range(N_SEEDS):
        algo_c = QIEA(problem, decode="permutation", seed=seed)
        F_control.append(algo_c.run(n_gen=N_GEN).F)

        algo_r = QIEA(problem, decode="permutation", seed=seed, **CANDIDATE)
        F_candidate.append(algo_r.run(n_gen=N_GEN).F)
    elapsed = time.time() - t0

    ref_point = np.max(np.vstack(F_control + F_candidate), axis=0) * 1.1
    hv = HV(ref_point=ref_point)
    hv_control = np.array([hv(F) for F in F_control])
    hv_candidate = np.array([hv(F) for F in F_candidate])

    w = wilcoxon_test(hv_candidate, hv_control)
    pct = (hv_candidate.mean() / hv_control.mean() - 1) * 100

    print(
        f"[{instance_name}] n={N_SEEDS} t={elapsed:.1f}s  "
        f"control={hv_control.mean():.4e}  candidate={hv_candidate.mean():.4e}  "
        f"pct={pct:+.1f}%  wilcoxon p={w['p_value']:.4g}"
    )
    return pd.DataFrame(
        [
            dict(
                instance=instance_name,
                hv_control_mean=hv_control.mean(),
                hv_control_std=hv_control.std(),
                hv_candidate_mean=hv_candidate.mean(),
                hv_candidate_std=hv_candidate.std(),
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
    df.to_csv(out_dir / f"tune_qiea_restart_confirm_{instance}.csv", index=False)
