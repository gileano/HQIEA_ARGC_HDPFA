"""
OFAT sweep of neighborhood_size / mutation_prob on the two larger CVRP instances
(E-n101-k8, route2_199), where QIEA remains weakest of the five algorithms even
after the theta_min tuning (section 8) and rotation-step scaling fix (section 9).
Both prior sweeps only tuned these two knobs on A-n32-k5 and found no improvement
there over the defaults -- but neither was ever re-run on the larger instances,
which is the open question section 9h flags as the next candidate lever.

n_partitions is held fixed at 5, matching MOEA/D's and RVEA's reference directions
in run_baselines.build_algorithms() (same fairness argument as tune_qiea_matched.py).
theta_max/theta_min are left at None (not passed), so QIEA._default_theta_bounds
resolves them via the section-9 gap-relative formula -- i.e. this sweep tunes on
top of the current best-known state, not the pre-section-9 fixed defaults.
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

FIXED_N_PARTITIONS = 5  # parity with MOEA/D + RVEA's ref_dirs in run_baselines.build_algorithms

DEFAULTS = dict(neighborhood_size=10, mutation_prob=0.05, n_partitions=FIXED_N_PARTITIONS)

SWEEP = {
    "neighborhood_size": [5, 10, 15, 20, 30, 50],
    "mutation_prob": [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
}

N_SEEDS = 5
N_GEN = 80
CONFIRM_SEEDS = 15


def parse_value(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def build_configs():
    configs = [("baseline", dict(DEFAULTS))]
    for param, values in SWEEP.items():
        for v in values:
            if v == DEFAULTS[param]:
                continue
            cfg = dict(DEFAULTS)
            cfg[param] = v
            configs.append((f"{param}={v}", cfg))
    return configs


def run_config(problem, config, n_seeds, n_gen):
    runs_F, runs_time = [], []
    for seed in range(n_seeds):
        t0 = time.time()
        algo = QIEA(problem, decode="permutation", seed=seed, **config)
        res = algo.run(n_gen=n_gen)
        runs_F.append(res.F)
        runs_time.append(time.time() - t0)
    return runs_F, runs_time


def indicator_rows(label, cfg, runs_F, runs_time, ref_point):
    rows = []
    for run_idx, F in enumerate(runs_F):
        if len(F) == 0:
            continue
        ind = all_indicators(F, ref_point)
        ind.update(config_label=label, run=run_idx, time_s=runs_time[run_idx], **cfg)
        rows.append(ind)
    return rows


def sweep_instance(instance, base, out_dir):
    inst = CVRPInstance.from_file(base / "data" / "processed" / f"{instance}.json")
    problem = CVRPProblem(inst)

    configs = build_configs()
    all_results = {}
    for label, cfg in configs:
        runs_F, runs_time = run_config(problem, cfg, N_SEEDS, N_GEN)
        all_results[label] = (cfg, runs_F, runs_time)
        mean_archive = np.mean([len(f) for f in runs_F])
        print(f"[{instance}] done: {label:24s}  mean_time={np.mean(runs_time):.2f}s  mean_archive={mean_archive:.1f}")

    ref_point = (
        np.max(
            np.vstack([f for (_, runs_F, _) in all_results.values() for f in runs_F if len(f) > 0]),
            axis=0,
        )
        * 1.1
    )

    rows = []
    for label, (cfg, runs_F, runs_time) in all_results.items():
        rows.extend(indicator_rows(label, cfg, runs_F, runs_time, ref_point))
    df = pd.DataFrame(rows)

    summary = df.groupby("config_label")["hypervolume"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"\n[{instance}] Hypervolume mean +- std per config, sorted best-first:")
    print(summary.round(4).to_string())

    baseline_hv = summary.loc["baseline", "mean"]
    print(f"[{instance}] baseline (matched n_partitions=5) mean HV over {N_SEEDS} seeds = {baseline_hv:.4f}")

    best_per_param = {}
    for param in SWEEP:
        sub = summary[summary.index.str.startswith(f"{param}=")]
        if len(sub) == 0:
            continue
        best_label = sub["mean"].idxmax()
        best_val = parse_value(best_label.split("=", 1)[1])
        beats_baseline = sub.loc[best_label, "mean"] > baseline_hv
        best_per_param[param] = best_val if beats_baseline else DEFAULTS[param]
        flag = "" if beats_baseline else "  (does NOT beat baseline -> keeping default)"
        print(f"  [{instance}] best {param}: {best_val} (mean HV {sub.loc[best_label, 'mean']:.4f} vs baseline {baseline_hv:.4f}){flag}")

    combined_cfg = dict(DEFAULTS)
    combined_cfg.update(best_per_param)
    print(f"[{instance}] combined-best config: {combined_cfg}")

    # Confirm combined-best vs baseline with more seeds + paired Wilcoxon (guards against
    # the OFAT interaction trap section 8c hit -- a naive per-param combination there was
    # WORSE than the unmodified baseline).
    confirm_baseline_F, confirm_baseline_time = run_config(problem, DEFAULTS, CONFIRM_SEEDS, N_GEN)
    confirm_combined_F, confirm_combined_time = run_config(problem, combined_cfg, CONFIRM_SEEDS, N_GEN)

    confirm_ref_point = (
        np.max(
            np.vstack([f for f in confirm_baseline_F + confirm_combined_F if len(f) > 0]),
            axis=0,
        )
        * 1.1
    )
    baseline_hv_confirm = np.array(
        [all_indicators(f, confirm_ref_point)["hypervolume"] for f in confirm_baseline_F if len(f) > 0]
    )
    combined_hv_confirm = np.array(
        [all_indicators(f, confirm_ref_point)["hypervolume"] for f in confirm_combined_F if len(f) > 0]
    )
    n = min(len(baseline_hv_confirm), len(combined_hv_confirm))
    stat, p = stats.wilcoxon(combined_hv_confirm[:n], baseline_hv_confirm[:n])
    pct_change = 100.0 * (combined_hv_confirm.mean() - baseline_hv_confirm.mean()) / baseline_hv_confirm.mean()
    print(
        f"[{instance}] CONFIRM ({CONFIRM_SEEDS} seeds): baseline HV={baseline_hv_confirm.mean():.4f}, "
        f"combined-best HV={combined_hv_confirm.mean():.4f} ({pct_change:+.1f}%), "
        f"Wilcoxon p={p:.4f} ({'significant' if p < 0.05 else 'not significant'})"
    )

    rows.extend(indicator_rows("confirm-baseline", DEFAULTS, confirm_baseline_F, confirm_baseline_time, confirm_ref_point))
    rows.extend(indicator_rows("confirm-combined-best", combined_cfg, confirm_combined_F, confirm_combined_time, confirm_ref_point))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"tune_qiea_large_{instance}.csv", index=False)

    return {
        "instance": instance,
        "combined_cfg": combined_cfg,
        "baseline_hv_mean": float(baseline_hv_confirm.mean()),
        "combined_hv_mean": float(combined_hv_confirm.mean()),
        "pct_change": float(pct_change),
        "wilcoxon_p": float(p),
    }


def main(instances):
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results"
    summaries = [sweep_instance(instance, base, out_dir) for instance in instances]
    print("\n=== SUMMARY ===")
    for s in summaries:
        sig = "significant" if s["wilcoxon_p"] < 0.05 else "not significant"
        print(
            f"{s['instance']:12s} combined-best={s['combined_cfg']}  "
            f"{s['pct_change']:+.1f}% HV vs baseline  (Wilcoxon p={s['wilcoxon_p']:.4f}, {sig})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*", default=["E-n101-k8", "route2_199"])
    args = parser.parse_args()
    main(args.instances)
