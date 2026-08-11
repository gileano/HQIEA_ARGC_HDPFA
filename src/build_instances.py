"""
Build many-objective CVRP instances from the Sibiu real-road distance matrices.

Input: paper2/data/raw/*.csv  (asymmetric distance matrices in km, Point_1 = depot)
Output: paper2/data/processed/*.json  (self-contained CVRP instance: distance matrix,
        synthesized demand, vehicle capacity, and per-edge time/cost/emission structure)

Synthesized-data assumptions (documented here because there is no real demand/capacity
source for these routes -- state these explicitly as a limitation in the paper):
  - demand[i]     ~ DiscreteUniform(40, 400) kg of waste per collection point, seeded.
  - road_factor_ij ~ LogNormal per directed edge, independent of distance: models
                     heterogeneous congestion/road type so that travel TIME is not a
                     scalar multiple of distance (otherwise time/distance/cost/emissions
                     would be perfectly collinear and the Pareto front degenerates to a
                     single point, as observed in the pymoo sanity check).
  - avg_speed_kmh  = 25 km/h base urban speed, scaled per edge by road_factor.
  - cost           = variable fuel/maintenance cost per km (COST_VAR_PER_KM) is itself
                     scaled by road_factor (congestion burns more fuel) PLUS a per-route
                     FIXED_COST_PER_VEHICLE -- this makes cost depend on how many vehicles
                     a solution uses, not just total distance, which is the real economic
                     trade-off in fleet sizing.
  - emissions      depend on both distance and time (idling in congestion emits CO2 even
                    without covering distance) AND on the truck's current payload, which
                    grows monotonically along a collection route (trucks fill up as they
                    collect, unlike delivery routes that empty out) -- so emissions are
                    a route-sequence-dependent quantity, computed at evaluation time in
                    problem.py, not a static matrix.
  - vehicle_capacity is sized so each instance needs roughly TARGET_VEHICLES routes.

These five knobs (distance, time, cost, emissions, workload balance) are the paper's
stated objective set. Distance and workload balance were already independent; this
version makes time/cost/emissions genuinely independent of distance too.
"""
import csv
import json
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

BASE_SPEED_KMH = 25.0
ROAD_FACTOR_SIGMA = 0.35  # log-normal spread of per-edge congestion/road-type factor

COST_VAR_PER_KM = 1.0
FIXED_COST_PER_VEHICLE = 40.0

EMISSION_PER_KM = 0.7  # kg CO2 per km, distance component
EMISSION_PER_HOUR_IDLE = 1.2  # kg CO2 per hour, congestion/idling component
EMISSION_LOAD_FACTOR = 0.004  # extra kg CO2 per km per kg of current payload

DEMAND_LOW, DEMAND_HIGH = 40, 400
TARGET_VEHICLES = 6
CAPACITY_SLACK = 1.15
SEED = 42

# SBxxSOM files identified by node count against the first paper's route1_334 / route2_199 / route3_202.
FILE_TO_ROUTE = {
    "SB25SOM_distance_matrix.csv": "route1_334",
    "SB30SOM_distance_matrix.csv": "route2_199",
    "SB45SOM_distance_matrix.csv": "route3_202",
}


def load_distance_matrix(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    labels = [r[0] for r in body]
    matrix = np.array([[float(v) for v in r[1:]] for r in body])
    assert matrix.shape[0] == matrix.shape[1] == len(labels)
    return labels, matrix


def synthesize_demand(n, seed):
    rng = np.random.default_rng(seed)
    demand = rng.integers(DEMAND_LOW, DEMAND_HIGH + 1, size=n)
    demand[0] = 0  # depot
    return demand


def synthesize_road_factors(n, seed):
    rng = np.random.default_rng(seed + 1)
    factors = rng.lognormal(mean=0.0, sigma=ROAD_FACTOR_SIGMA, size=(n, n))
    np.fill_diagonal(factors, 1.0)
    return factors


def build_instance(csv_path, route_name):
    labels, dist = load_distance_matrix(csv_path)
    n = len(labels)
    demand = synthesize_demand(n, SEED)
    road_factor = synthesize_road_factors(n, SEED)

    total_demand = int(demand.sum())
    capacity = int(np.ceil(total_demand / TARGET_VEHICLES * CAPACITY_SLACK))

    speed_matrix = BASE_SPEED_KMH / road_factor
    time_matrix = dist / speed_matrix
    cost_matrix = dist * COST_VAR_PER_KM * road_factor

    instance = {
        "name": route_name,
        "source_file": csv_path.name,
        "num_points": n,
        "depot_index": 0,
        "labels": labels,
        "distance_matrix_km": dist.tolist(),
        "time_matrix_h": time_matrix.tolist(),
        "cost_matrix_variable": cost_matrix.tolist(),
        "road_factor": road_factor.tolist(),
        "demand": demand.tolist(),
        "vehicle_capacity": capacity,
        "target_vehicles": TARGET_VEHICLES,
        "assumptions": {
            "base_speed_kmh": BASE_SPEED_KMH,
            "road_factor_sigma": ROAD_FACTOR_SIGMA,
            "cost_var_per_km": COST_VAR_PER_KM,
            "fixed_cost_per_vehicle": FIXED_COST_PER_VEHICLE,
            "emission_per_km": EMISSION_PER_KM,
            "emission_per_hour_idle": EMISSION_PER_HOUR_IDLE,
            "emission_load_factor": EMISSION_LOAD_FACTOR,
            "demand_range": [DEMAND_LOW, DEMAND_HIGH],
            "capacity_slack": CAPACITY_SLACK,
            "seed": SEED,
        },
    }
    return instance


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, route_name in FILE_TO_ROUTE.items():
        csv_path = RAW_DIR / fname
        if not csv_path.exists():
            print(f"skip (missing): {csv_path}")
            continue
        inst = build_instance(csv_path, route_name)
        out_path = OUT_DIR / f"{route_name}.json"
        with open(out_path, "w") as f:
            json.dump(inst, f)
        print(
            f"{route_name}: n={inst['num_points']} "
            f"total_demand={sum(inst['demand'])} capacity={inst['vehicle_capacity']} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
