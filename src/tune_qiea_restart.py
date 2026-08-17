"""
Screen for the plateau-gated diversity-reinjection mechanism added to qiea.py
(restart_patience/restart_fraction, default off -- see paper1.txt section 14/15).
Directly tests whether reseeding a fraction of subproblems once the archive has
stopped growing lets QIEA break past the hard hypervolume plateau found by
diag_qiea_generation_trace.py on A-n32-k5 (freezes ~gen 60) and route2_199
(freezes ~gen 440-450), at the paper's actual n-gen=500 budget.

Grid: restart_patience in {20, 50, 100, 200} x restart_fraction in {0.1, 0.3, 0.5},
plus a restart-disabled control (patience=None, i.e. current default/shipped
behavior), on A-n32-k5, E-n101-k8, route2_199. n_partitions=5 (matched to
MOEA/D+RVEA for fairness), theta/mutation bounds left on their auto-scaled
defaults. Own reference point per instance (max over all configs run here * 1.1)
-- NOT comparable in absolute terms to run_experiment.py's or other tune_*.py
scripts' hypervolume numbers, same convention as every prior tune_qiea_*.py.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402

N_GEN = 500
N_SEEDS = 5
PATIENCE_GRID = [20, 50, 100, 200]
FRACTION_GRID = [0.1, 0.3, 0.5]
INSTANCES = ["A-n32-k5", "E-n101-k8", "route2_199"]


def configs():
    yield ("control", None, None)
    for p in PATIENCE_GRID:
        for f in FRACTION_GRID:
            yield (f"p{p}_f{f}", p, f)


def run_instance(instance_name):
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{instance_name}.json"
    inst = CVRPInstance.from_file(inst_path)
    problem = CVRPProblem(inst)

    all_F = {}
    n_restarts = {}
    t0 = time.time()
    for label, patience, fraction in configs():
        Fs, restarts = [], []
        for seed in range(N_SEEDS):
            kwargs = dict(decode="permutation", seed=seed)
            if patience is not None:
                kwargs["restart_patience"] = patience
                kwargs["restart_fraction"] = fraction
            algo = QIEA(problem, **kwargs)
            res = algo.run(n_gen=N_GEN)
            Fs.append(res.F)
            restarts.append(algo.n_restarts)
        all_F[label] = Fs
        n_restarts[label] = np.mean(restarts)
    print(f"[{instance_name}] all configs done in {time.time() - t0:.1f}s")

    ref_point = np.max(np.vstack([f for runs in all_F.values() for f in runs]), axis=0) * 1.1
    hv = HV(ref_point=ref_point)

    rows = []
    for label in all_F:
        hvs = [hv(F) for F in all_F[label]]
        rows.append(
            dict(
                instance=instance_name,
                config=label,
                hv_mean=np.mean(hvs),
                hv_std=np.std(hvs),
                mean_restarts=n_restarts[label],
            )
        )
    df = pd.DataFrame(rows).sort_values("hv_mean", ascending=False)
    control_hv = df.loc[df["config"] == "control", "hv_mean"].iloc[0]
    df["pct_vs_control"] = (df["hv_mean"] / control_hv - 1) * 100
    return df


if __name__ == "__main__":
    all_rows = []
    for inst in INSTANCES:
        df = run_instance(inst)
        print(df.to_string(index=False))
        print()
        all_rows.append(df)
    pd.concat(all_rows).to_csv(
        Path(__file__).resolve().parent.parent / "results" / "tune_qiea_restart_screen.csv", index=False
    )
