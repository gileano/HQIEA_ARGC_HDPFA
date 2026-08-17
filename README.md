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

For the permutation decode, both the rotation-step size and the mutation rate default to a
function of instance size (`n_var`) rather than a fixed constant — a fixed absolute value
tuned on a small instance becomes disproportionately disruptive to `argsort`-decoded tours as
the instance grows, which was quietly capping QIEA's performance on larger CVRP instances (see
`paper1.txt` sections 9-10).

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

**Full-scale campaign now run (2026-08-17)** — `results/*_indicators.csv` and
`*_fronts.npz` hold the paper's actual target-scale numbers (30 runs, `--n-gen 500`,
`--pop-size 80`) on all 7 instances, not pilot-scale data. Result: **QIEA is the weakest
of all five algorithms on every single instance** (hypervolume ratio to the best baseline
ranges 0.17–0.45 across the 7 instances; Friedman p ≪ 0.05 on all 7). This contradicts an
earlier pilot-scale finding (3–10 seeds, `--n-gen 80`) that mutation-rate scaling had made
QIEA competitive on the two larger instances tested at the time — that result turned out
to be an artifact of the short generation budget, not a real fix (see `paper1.txt`
section 14).

A follow-up diagnostic (`src/diag_qiea_generation_trace.py`, no tuning — just traces
hypervolume every 10 generations) found the actual cause: **QIEA hits a hard hypervolume
plateau early in the run and then never improves again**, while every baseline keeps
climbing steadily all the way to generation 500.

- A-n32-k5: QIEA is competitive through ~gen 30–60, then freezes bit-for-bit at gen 60
  for the remaining 440 generations, while NSGA-II/SPEA2/RVEA climb another ~2.5–3x past
  that point.
- route2_199: same shape, plateau starts ~gen 440–450; final QIEA hypervolume is under
  half of the next-worst baseline.

This is a **structural issue, not a parameter-tuning gap** — the auto-scaled
`theta_min`/`theta_max`/`mutation_prob` formulas from the earlier tuning passes (below)
were already active during this trace. Root cause: QIEA's fixed-size population (one
individual per MOEA/D subproblem) converges to its neighborhood guides, and the
diversity-stagnation escape — "ARGC", the algorithm's namesake mechanism — essentially
never fires under its current threshold (confirmed separately in `paper1.txt` section 12).
With no escape actually engaging, there's no way to find new archive points once
converged, so a longer generation budget helps every other algorithm but does nothing for
QIEA past its freeze point.

**Current top priority** (supersedes prior tuning-focused next steps): design and test a
diversity-reinjection or plateau-gated restart mechanism, and confirm the plateau
generalizes to the other 5 instances (only A-n32-k5 and route2_199 traced so far). See
`paper1.txt` section 14g for the full open-questions list.

<details>
<summary>Earlier tuning passes (2026-08-12/13, superseded by the finding above)</summary>

Three tuning passes (`paper1.txt` sections 8–10), all under a strict population-fairness
constraint (QIEA's population size, `n_partitions`, kept identical to MOEA/D's and
RVEA's in every comparison):

- `theta_min`/`theta_max` (rotation-gate step size) made instance-size-aware — a fixed
  absolute value tuned on the smallest instance (A-n32-k5, 32 nodes) was an increasingly
  disruptive fraction of the `argsort` gap as instance size grew.
- `mutation_prob` made instance-size-aware the same way — a fixed mutation rate flips a
  growing *number* of genes as `n_var` grows, which was quietly crippling QIEA on the
  larger instances tested.

These looked like real fixes in 10-seed pilot reruns at `--n-gen 80` (QIEA statistically
tied with NSGA-II/SPEA2/MOEA-D on the two larger instances tested) — but the full-scale
campaign above shows that result did not hold once run at the paper's actual generation
budget. `neighborhood_size` and the diversity-stagnation-boost parameters were also
investigated and left unchanged (real, reproducible effects too instance-specific or
counterproductive to ship as defaults — see `paper1.txt` sections 11–13).

</details>

See `paper1.txt` in the parent directory for the full project plan and implementation log.
