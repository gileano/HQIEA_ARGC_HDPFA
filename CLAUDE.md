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
`qiea.py`) decays linearly over generations (explore -> exploit), but is boosted 3x whenever
angular diversity (`diversity()`) stalls over a sliding window (`diversity_window`,
`diversity_stagnation_tol`) — together with a temporary mutation-probability bump. This
stagnation-escape logic is the "ARGC" in the project name.

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

## Status / known limitations

Current results (see README "Status") are pilot-scale (`--n-gen 60-80 --n-runs 3-5`), not the
paper's final numbers. QIEA is currently on par with NSGA-II/SPEA2 but behind MOEA/D and RVEA
on hypervolume — it is un-tuned, not necessarily worse in principle. Do not present pilot-run
numbers as final results. The Sibiu file-to-route-count mapping (`SB25SOM`/`SB30SOM`/`SB45SOM`
-> `route1_334`/`route2_199`/`route3_202`) is unconfirmed against original records — flag this
if it becomes load-bearing for a claim.
