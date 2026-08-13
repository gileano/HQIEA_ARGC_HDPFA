# HQIEA-ARGC-HDPFA

A Hybrid Quantum-Inspired Evolutionary Algorithm with Adaptive Rotation-Gate Control for
High-Dimensional Pareto-Front Approximation, applied to a real-world many-objective
Capacitated Vehicle Routing Problem (CVRP): urban waste collection in Sibiu, Romania.

This is the code for a second paper extending a prior single-objective QIEA/TSP study
(qubit-chromosome + rotation-gate operator) to a many-objective setting: 5 conflicting
objectives (distance, time, cost, emissions, workload balance), validated on standard
synthetic suites (ZDT/DTLZ/WFG), CVRPLIB instances, and real Sibiu routing data, against
NSGA-II, SPEA2, MOEA/D and RVEA baselines.

## Key idea

The chromosome is a vector of qubit angles `theta_i in [0, pi/2]` (real-amplitude qubit
state). Two decode modes share this representation:

- **Permutation** (CVRP): `tour = argsort(theta)`. Sorting real numbers can never produce
  a duplicate, so there is **no repair step** — this directly fixes the diversity collapse
  reported for the qubit chromosome in the first (single-objective) paper at high city counts.
- **Continuous** (ZDT/DTLZ/WFG): `x_i = xl_i + sin(theta_i)^2 * (xu_i - xl_i)`.

Scalability to many objectives comes from an MOEA/D-style decomposition (Das-Dennis weight
vectors, Tchebycheff scalarization, neighborhood-restricted mating). The rotation-gate step
size decays over generations but is boosted when population diversity stalls, giving an
adaptive explore/exploit schedule.

## Problem formulation

Decision variable: a giant-tour permutation of customers, decoded into vehicle routes by a
capacity-first greedy split (feasible by construction — no penalty terms needed). Five
objectives, all minimized:

| Objective | Unit | Notes |
|---|---|---|
| Distance | km | sum of edge distances, depot round-trips included |
| Time | h | per-edge road factor makes this independent of distance (congestion/road type) |
| Cost | currency units | fixed cost per vehicle + variable fuel cost |
| Emissions | kg CO2 | distance + idling(time) + payload-dependent term (trucks fill up while collecting) |
| Workload balance | — | std. dev. of per-route completion time across vehicles used |

Full detail and exact formulas: `src/problem.py` docstring.

## Data

- `data/raw/` — original Sibiu real-road distance matrices (`SB25SOM`/`SB30SOM`/`SB45SOM`),
  mapped by node count to `route1_334` / `route2_199` / `route3_202` from the first paper.
  **This size-based mapping is not yet confirmed against original records.**
- `data/cvrplib_raw/` — downloaded CVRPLIB instances (`A-n32-k5`, `A-n33-k5`, `E-n101-k8`,
  `M-n200-k16`) in TSPLIB-CVRP format.
- `data/processed/` — everything converted to one shared JSON schema. Demand, vehicle
  capacity, per-edge road factor, and cost/emission coefficients are **synthesized** with a
  fixed seed (no real demand/capacity data exists for the Sibiu routes) — see
  `src/build_instances.py` for exact distributions. This is a documented assumption/limitation,
  not real municipal data.

## Repository structure

```
HQIEA_ARGC_HDPFA/
├── data/
│   ├── raw/                 Sibiu distance-matrix CSVs
│   ├── cvrplib_raw/         downloaded CVRPLIB .vrp files
│   └── processed/           final JSON instances used by every script
├── src/
│   ├── build_instances.py           Sibiu CSV -> JSON instance
│   ├── build_cvrplib_instances.py   CVRPLIB .vrp -> JSON instance (same schema)
│   ├── problem.py                   CVRP model: decode, evaluate, pymoo Problem wrapper
│   ├── qiea.py                      the QIEA itself
│   ├── run_baselines.py             NSGA-II / SPEA2 / MOEA-D / RVEA via pymoo
│   ├── run_synthetic.py             QIEA vs NSGA-II on ZDT/DTLZ/WFG
│   ├── run_experiment.py            full comparison + indicators + Wilcoxon/Friedman
│   ├── metrics.py                   hypervolume / spacing / spread / IGD / IGD+ / stats tests
│   ├── ortools_sanity.py            near-optimal distance reference (OR-Tools)
│   └── sanity_check.py              first smoke test, superseded by problem.py
├── results/                 per-run indicator CSVs and raw Pareto fronts (.npz)
└── requirements.txt
```

## Setup

```bash
git clone git@github.com:gileano/HQIEA_ARGC_HDPFA.git
cd HQIEA_ARGC_HDPFA

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12. Core dependencies: `pymoo`, `deap`, `jmetalpy`, `ortools`, `numpy`,
`pandas`, `scipy`, `statsmodels`.

## Usage

All commands below assume `cd HQIEA_ARGC_HDPFA && source .venv/bin/activate`.

**Regenerate instances** (only needed if raw CSV/`.vrp` files change):

```bash
python src/build_instances.py
python src/build_cvrplib_instances.py
```

**Quick sanity checks** (seconds):

```bash
python src/sanity_check.py     # pymoo + Sibiu data pipeline smoke test
python src/problem.py          # decode/evaluate smoke test on route2_199
python src/qiea.py             # QIEA alone on route2_199
python src/ortools_sanity.py   # near-optimal distance reference on A-n32-k5
```

**Synthetic benchmark comparison** — QIEA vs NSGA-II on ZDT1-3 / DTLZ1-2 / WFG1:

```bash
python src/run_synthetic.py
```

**Baselines only**, on one instance:

```bash
python -c "from pathlib import Path; from src.run_baselines import run_all; \
            run_all(Path('data/processed/route2_199.json'), n_gen=50, pop_size=80)"
```

**Full comparison** — QIEA + all 4 baselines, N seeds, indicators, Wilcoxon + Friedman:

```bash
python src/run_experiment.py <instance_name> --n-gen <G> --n-runs <N> --pop-size <P>

# example
python src/run_experiment.py route2_199 --n-gen 100 --n-runs 5 --pop-size 80
```

Available `<instance_name>` values: `route1_334`, `route2_199`, `route3_202`, `A-n32-k5`,
`A-n33-k5`, `E-n101-k8`, `M-n200-k16`.

Writes `results/<instance_name>_indicators.csv` (hypervolume/spacing/spread per run) and
`results/<instance_name>_fronts.npz` (raw objective-space fronts).

## Status

Pilot-scale results only (`--n-gen 60-80 --n-runs 3-5`) — not the paper's final numbers.
A first hyperparameter tuning pass (2026-08-12, see `paper1.txt` section 8) raised
`theta_min` from 0.02 to 0.1 in `qiea.py` after a population-fairness check and a
factorial re-verification. This measurably helps on the smallest instance (A-n32-k5:
QIEA now beats NSGA-II/SPEA2, still behind MOEA/D/RVEA) but does **not** generalize to
the larger instances tested (E-n101-k8, route2_199: QIEA remains weakest of the five).
Friedman tests confirm the cross-algorithm differences are statistically significant on
every instance. The algorithm produces valid, non-degenerate, well-spread fronts; the
open question is now scaling the tuning (or the algorithm itself) to larger instances,
not "it hasn't been tuned yet."

See `paper1.txt` in the parent directory for the full project plan and implementation log.
