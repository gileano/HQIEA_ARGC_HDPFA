"""
Parse CVRPLIB .vrp files (TSPLIB-style CVRP format) and convert them into the
same 5-objective JSON schema used for the Sibiu instances, so the QIEA and all
baselines run through one code path (problem.py) regardless of data source.

CVRPLIB only defines coordinates/demand/capacity (i.e. the classic single-objective
CVRP). To get genuinely comparable many-objective results across CVRPLIB and Sibiu,
the same synthetic time/cost/emission generation procedure from build_instances.py
(per-edge road factor, fixed+variable cost, load-dependent emissions) is applied here
too, with the same documented assumptions and seed.
"""
import re
from pathlib import Path

import numpy as np

from build_instances import (
    BASE_SPEED_KMH,
    CAPACITY_SLACK,
    EMISSION_LOAD_FACTOR,
    EMISSION_PER_HOUR_IDLE,
    EMISSION_PER_KM,
    FIXED_COST_PER_VEHICLE,
    ROAD_FACTOR_SIGMA,
    COST_VAR_PER_KM,
    SEED,
    synthesize_road_factors,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "cvrplib_raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def parse_vrp(path):
    text = path.read_text()
    name = re.search(r"NAME\s*:\s*(\S+)", text).group(1)
    capacity = int(re.search(r"CAPACITY\s*:\s*(\d+)", text).group(1))

    coords = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.\-]+)\s+([\d.\-]+)\s*$", text, re.MULTILINE):
        idx, x, y = m.groups()
        coords[int(idx)] = (float(x), float(y))

    demand_section = text.split("DEMAND_SECTION")[1].split("DEPOT_SECTION")[0]
    demand = {}
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s*$", demand_section, re.MULTILINE):
        idx, d = m.groups()
        demand[int(idx)] = int(d)

    ids = sorted(coords.keys())
    n = len(ids)
    xy = np.array([coords[i] for i in ids])
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    demand_arr = np.array([demand[i] for i in ids], dtype=float)

    return name, n, dist, demand_arr, capacity


def build_instance(vrp_path):
    name, n, dist, demand, capacity = parse_vrp(vrp_path)
    road_factor = synthesize_road_factors(n, SEED)

    speed_matrix = BASE_SPEED_KMH / road_factor
    time_matrix = dist / speed_matrix
    cost_matrix = dist * COST_VAR_PER_KM * road_factor

    instance = {
        "name": name,
        "source_file": vrp_path.name,
        "num_points": n,
        "depot_index": 0,
        "labels": [f"Point_{i+1}" for i in range(n)],
        "distance_matrix_km": dist.tolist(),
        "time_matrix_h": time_matrix.tolist(),
        "cost_matrix_variable": cost_matrix.tolist(),
        "road_factor": road_factor.tolist(),
        "demand": demand.tolist(),
        "vehicle_capacity": float(capacity),
        "target_vehicles": None,
        "assumptions": {
            "base_speed_kmh": BASE_SPEED_KMH,
            "road_factor_sigma": ROAD_FACTOR_SIGMA,
            "cost_var_per_km": COST_VAR_PER_KM,
            "fixed_cost_per_vehicle": FIXED_COST_PER_VEHICLE,
            "emission_per_km": EMISSION_PER_KM,
            "emission_per_hour_idle": EMISSION_PER_HOUR_IDLE,
            "emission_load_factor": EMISSION_LOAD_FACTOR,
            "capacity_slack": CAPACITY_SLACK,
            "seed": SEED,
            "note": "distance/demand/capacity are the original CVRPLIB values; "
            "time/cost/emissions are synthesized with the same procedure as the Sibiu instances.",
        },
    }
    return instance


def main():
    import json

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vrp_path in sorted(RAW_DIR.glob("*.vrp")):
        inst = build_instance(vrp_path)
        out_path = OUT_DIR / f"{inst['name']}.json"
        with open(out_path, "w") as f:
            json.dump(inst, f)
        print(f"{inst['name']}: n={inst['num_points']} capacity={inst['vehicle_capacity']} -> {out_path}")


if __name__ == "__main__":
    main()
