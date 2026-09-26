# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper: a Hybrid Quantum-Inspired Evolutionary Algorithm with Adaptive
Rotation-Gate Control (HQIEA-ARGC) for many-objective optimization, applied to a 5-objective
Capacitated Vehicle Routing Problem (urban waste collection, Sibiu, Romania), plus synthetic
ZDT/DTLZ/WFG benchmarks. It is compared against NSGA-II, SPEA2, MOEA/D, and RVEA (all via pymoo).

This extends a prior single-objective QIEA/TSP paper — the key design decisions in this repo
(no-repair permutation decode, MOEA/D decomposition, diversity-triggered rotation boost) exist
specifically to fix scalability/diversity problems identified in that earlier work. When
touching `qiea.py` or `problem.py`, the module docstrings explain *why* each mechanism exists —
read them before changing the encoding or decode logic.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12. No lint config, formatter config, or test suite exists in this repo —
don't invent `pytest`/`ruff`/`black` invocations.

## Common commands

All commands assume `source .venv/bin/activate` from the repo root.

```bash
# Regenerate data/processed/*.json (only needed if raw CSV/.vrp files change)
python src/build_instances.py
python src/build_cvrplib_instances.py

# Quick smoke tests (seconds)
python src/sanity_check.py     # pymoo + Sibiu data pipeline
python src/problem.py          # decode/evaluate on route2_199
python src/qiea.py             # QIEA alone on route2_199
python src/ortools_sanity.py   # near-optimal distance reference on A-n32-k5

# QIEA vs NSGA-II on ZDT1-3 / DTLZ1-2 / WFG1
python src/run_synthetic.py

# Baselines only, one instance
python -c "from pathlib import Path; from src.run_baselines import run_all; \
            run_all(Path('data/processed/route2_199.json'), n_gen=50, pop_size=80)"

# Full comparison: QIEA + 4 baselines, N seeds, indicators, Wilcoxon + Friedman
python src/run_experiment.py <instance_name> --n-gen <G> --n-runs <N> --pop-size <P>
# e.g. python src/run_experiment.py route2_199 --n-gen 100 --n-runs 5 --pop-size 80
```

`<instance_name>`: `route1_334`, `route2_199`, `route3_202`, `A-n32-k5`, `A-n33-k5`,
`E-n101-k8`, `M-n200-k16`. `run_experiment.py` writes `results/<instance>_indicators.csv`
(hypervolume/spacing/spread per run) and `results/<instance>_fronts.npz` (raw fronts).
`run_experiment.py` deliberately takes `--n-gen`/`--n-runs` as required-ish CLI args rather
than hardcoding paper-scale defaults (30 runs x 500-1000 gens) — pick the budget consciously.

## Architecture

**Shared representation (`qiea.py`):** every individual is a vector of qubit angles
`theta in [0, pi/2]`. Two decode modes read the same vector:
- `permutation` (CVRP): `tour = argsort(theta)` — sorting reals can't produce duplicates, so
  there is no repair step. This is the direct fix for the diversity collapse the first paper
  hit at high city counts.
- `continuous` (ZDT/DTLZ/WFG): `x_i = xl_i + sin(theta_i)^2 * (xu_i - xl_i)`.

**Many-objective scalability:** `QIEA` uses MOEA/D-style decomposition — Das-Dennis weight
vectors partition objective space into H Tchebycheff subproblems, one individual per
subproblem, mating restricted to each subproblem's nearest-neighbor set (`neighborhood_size`).

**Adaptive rotation gate:** the per-generation rotation step size (`rotation_angle` in
`qiea.py`) decays linearly over generations (explore -> exploit), but is boosted
(`rotation_boost_multiplier`, default 3x) whenever angular diversity (`diversity()`) stalls over
a sliding window (`diversity_window`, `diversity_stagnation_tol`) — together with a temporary
mutation-probability bump (`mutation_boost_multiplier`, default 5x). This stagnation-escape
logic is the "ARGC" in the project name. In practice it almost never fires at the default
tolerance, and loosening the tolerance made results worse. See `logs.txt` sections 12–13.

**Plateau-gated restart:** once the archive hasn't grown for `restart_patience` (default 10)
generations, `_do_restart` reseeds `restart_fraction` (default 0.5) of the H subproblems with
fresh random theta; the rest stay untouched, so good solutions are kept. This fixed an early
hard hypervolume plateau and is the largest validated improvement in the project
(`logs.txt` section 15). `restart_patience=None` disables it.

**Size-scaled defaults (permutation decode only):** `theta_min`/`theta_max`/`mutation_prob`
default to `None` and resolve via `_default_theta_bounds` / `_default_mutation_prob`, which
shrink them as `n_var` grows. They are clamped so they reduce *exactly* to the old
hand-tuned values at `n_var <= 31` — this representation is sensitive to ~1% drift there
(`logs.txt` section 9e). Continuous decode keeps fixed values.

**Fairness rule for defaults:** QIEA, MOEA/D and RVEA share `n_partitions=5` (H=126).
Never ship per-instance hyperparameters, or change QIEA's H without changing the
baselines' too — either would invalidate the comparison. This is why `neighborhood_size`
stays at 10 even though it helps route2_199.

**Problem layer (`problem.py`):** `CVRPInstance` holds precomputed distance/time/cost matrices
and evaluation logic; `CVRPProblem` wraps it as a pymoo `Problem` (n_obj=5) so QIEA and every
pymoo baseline (NSGA-II/SPEA2/MOEA-D/RVEA in `run_baselines.py`) evaluate identically. A
permutation decodes into vehicle routes via a capacity-first greedy split — feasible by
construction, so no repair step or penalty term is needed anywhere in the pipeline. The 5
objectives (distance, time, cost, emissions, workload balance) and their exact formulas are
documented in this file's module docstring; emissions has a payload-dependent term because
trucks fill up while collecting (order-dependent, not just distance-dependent).

**Data flow:** `data/raw/` (Sibiu CSV distance matrices) and `data/cvrplib_raw/` (CVRPLIB
`.vrp` files) are both converted by `build_instances.py` / `build_cvrplib_instances.py` into
one shared JSON schema in `data/processed/` — this is the only format every script reads.
Demand, vehicle capacity, per-edge road factor, and cost/emission coefficients for the Sibiu
instances are *synthesized* with a fixed seed (no real municipal demand data exists) — treat
these as a documented modeling assumption, not ground truth, when interpreting results.

**Evaluation layer (`metrics.py`):** hypervolume/IGD/IGD+/spacing come from pymoo; `spread`
(Deb's Delta) is hand-implemented since pymoo lacks it. IGD/IGD+ are only meaningful when a
true Pareto front exists (the synthetic suite) — real CVRP instances have no known true front,
so `run_experiment.py` scores them via hypervolume/spacing/spread against a shared reference
point instead. `wilcoxon_test` (paired, two algorithms) and `friedman_test` (many algorithms)
back the statistical-significance claims in the paper.

## Tuning / diagnostic scripts

`src/tune_*.py` (hyperparameter screens and confirm runs) and `src/diag_*.py` (read-only
diagnostics, e.g. `diag_qiea_generation_trace.py` for hypervolume-vs-generation traces) back
the findings recorded in `logs.txt`. Each script computes its own reference point, so
hypervolume values are only comparable *within* one script's output, not across scripts.
Lesson learned repeatedly: 5-seed screens on noisy grids often don't survive a 15–20-seed
confirmation — always confirm before changing a default.

## Status / known limitations

`logs.txt` is the authoritative research log (latest: section 16, 2026-08-18); read it before
proposing new tuning work so you don't repeat investigations that are already closed.

`results/*_indicators.csv` / `*_fronts.npz` now hold the **full-scale** campaign (30 runs,
n-gen=500, pop-size=80, all 7 instances, restart fix included). At that scale QIEA is
**last of five on every instance** (hypervolume 0.30–0.58x the best baseline); its only
statistical tie is with NSGA-II on route1_334. MOEA/D and RVEA are the hardest baselines.
Earlier pilot-scale claims (n-gen=80) that QIEA beat or tied NSGA-II/SPEA2 did **not** hold
at 500 generations — never cite pilot numbers as results.

Ruled out as explanations for the remaining gap: rotation-step size, mutation rate,
neighborhood size, stagnation-boost tolerance/magnitude, and population size H (varied
for all five algorithms together). Still open: continuous decode (ZDT/DTLZ/WFG) has never
been rechecked for the plateau or the scaling findings; the restart's runtime overhead
isn't measured; next idea is structural (MOEA/D and RVEA's reference-vector-guided
selection vs QIEA's Tchebycheff-neighborhood mating), not another hyperparameter sweep.
`run_synthetic.py` only compares QIEA vs NSGA-II on ZDT1-3/DTLZ1-2/WFG1. The QAOA baseline
from the paper plan has not been built.

The Sibiu file-to-route-count mapping (`SB25SOM`/`SB30SOM`/`SB45SOM`
-> `route1_334`/`route2_199`/`route3_202`) is unconfirmed against original records — flag this
if it becomes load-bearing for a claim.
