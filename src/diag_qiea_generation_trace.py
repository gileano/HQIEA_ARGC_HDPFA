"""
Diagnostic (no tuning, no source changes): trace hypervolume-vs-generation for QIEA
and the four pymoo baselines over a full 500-generation run, to check whether QIEA
plateaus early while the baselines keep improving -- the leading hypothesis for why
QIEA went from "ties NSGA-II/SPEA2" at n-gen=80 pilots to "weakest of five on every
instance" at the paper's actual 30-run/n-gen=500 scale (see logs.txt section 14
investigation this diagnostic feeds).

Reference point for hypervolume is fixed per (instance, seed) run: 1.1x the nadir of
the initial random population's objective values, shared by every algorithm and every
generation checkpoint in that run, so the HV values are comparable across generations
and across algorithms within one run. This is NOT the same ref_point convention as
run_experiment.py (which uses the max front value across all algorithms' FINAL fronts)
-- absolute HV numbers here are only meaningful as within-run trajectories, not as
cross-script comparisons.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance, CVRPProblem  # noqa: E402
from qiea import QIEA  # noqa: E402
from run_baselines import build_algorithms  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402

CHECKPOINT_EVERY = 10


def trace_qiea(problem, n_gen, seed, ref_point):
    algo = QIEA(problem, decode="permutation", seed=seed)
    hv = HV(ref_point=ref_point)
    _, F0 = algo.evaluate_population(algo.theta)
    algo.F = F0
    algo.z_ideal = np.minimum(algo.z_ideal, F0.min(axis=0))
    algo._update_archive(algo.theta, F0)

    trace = []
    for gen in range(n_gen):
        step, boosted = algo.rotation_angle(gen, n_gen)
        mut_p = algo.mutation_prob * (algo.mutation_boost_multiplier if boosted else 1.0)
        new_theta = np.empty_like(algo.theta)
        for i in range(algo.H):
            nbrs = algo.neighbors[i]
            from qiea import tchebycheff

            scal = [tchebycheff(algo.F[j], algo.weights[i], algo.z_ideal) for j in nbrs]
            guide = nbrs[int(np.argmin(scal))]
            mate = nbrs[algo.rng.integers(len(nbrs))]
            child = algo.quantum_crossover(algo.theta[i], algo.theta[mate])
            child = algo.rotate_toward(child, algo.theta[guide], step)
            child = algo.quantum_mutation(child, mut_p)
            new_theta[i] = child

        _, new_F = algo.evaluate_population(new_theta)
        algo.z_ideal = np.minimum(algo.z_ideal, new_F.min(axis=0))
        for i in range(algo.H):
            for j in algo.neighbors[i]:
                from qiea import tchebycheff

                if tchebycheff(new_F[i], algo.weights[j], algo.z_ideal) <= tchebycheff(
                    algo.F[j], algo.weights[j], algo.z_ideal
                ):
                    algo.theta[j] = new_theta[i]
                    algo.F[j] = new_F[i]

        algo.history_diversity.append(algo.diversity())
        algo._update_archive(algo.theta, algo.F)

        if (gen + 1) % CHECKPOINT_EVERY == 0 or gen == n_gen - 1:
            trace.append((gen + 1, float(hv(np.array(algo.archive_F)))))
    return trace


def trace_baseline(problem, algo_builder, n_gen, seed, ref_point):
    algo = algo_builder()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False, save_history=True)
    hv = HV(ref_point=ref_point)
    trace = []
    for h in res.history:
        gen = h.n_gen
        if gen % CHECKPOINT_EVERY == 0 or gen == n_gen:
            F = h.opt.get("F")
            trace.append((gen, float(hv(F))))
    return trace


def estimate_ref_point(problem, seed, margin=1.1):
    algo = QIEA(problem, decode="permutation", seed=seed)
    _, F0 = algo.evaluate_population(algo.theta)
    return F0.max(axis=0) * margin


def run_instance(instance_name, n_gen=500, seed=1):
    inst_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{instance_name}.json"
    inst = CVRPInstance.from_file(inst_path)
    problem = CVRPProblem(inst)
    ref_point = estimate_ref_point(problem, seed)

    print(f"\n=== {instance_name}  (n_gen={n_gen}, seed={seed}, ref_point={np.round(ref_point, 1)}) ===")

    t0 = time.time()
    qiea_trace = trace_qiea(problem, n_gen, seed, ref_point)
    print(f"QIEA      done in {time.time() - t0:6.1f}s")

    baselines = build_algorithms(problem.n_obj, pop_size=80)
    traces = {"QIEA": qiea_trace}
    for name in ["NSGA-II", "SPEA2", "MOEA/D", "RVEA"]:
        t0 = time.time()
        builder = lambda name=name: build_algorithms(problem.n_obj, pop_size=80)[name]
        traces[name] = trace_baseline(problem, builder, n_gen, seed, ref_point)
        print(f"{name:10s} done in {time.time() - t0:6.1f}s")

    header = "gen," + ",".join(traces.keys())
    print(header)
    n_checkpoints = len(qiea_trace)
    for k in range(n_checkpoints):
        row = [str(traces["QIEA"][k][0])]
        for name in traces:
            row.append(f"{traces[name][k][1]:.6e}" if k < len(traces[name]) else "")
        print(",".join(row))

    # plateau check: generation after which QIEA's HV never improves by >0.5% of final value again
    qiea_hvs = np.array([hv for _, hv in qiea_trace])
    final_hv = qiea_hvs[-1]
    tol = 0.005 * final_hv
    plateau_gen = None
    for k in range(len(qiea_hvs)):
        if np.all(qiea_hvs[k:] - qiea_hvs[k] <= tol):
            plateau_gen = qiea_trace[k][0]
            break
    print(f"QIEA plateau (no >0.5% improvement after this generation): {plateau_gen}")

    return traces


if __name__ == "__main__":
    for inst_name in ["A-n32-k5", "route2_199"]:
        run_instance(inst_name, n_gen=500, seed=1)
