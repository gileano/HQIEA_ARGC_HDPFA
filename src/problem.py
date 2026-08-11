"""
Many-objective CVRP formulation shared by the QIEA and all baselines.

Decision representation: a giant tour, i.e. a permutation of the n-1 customer
indices (depot excluded). It is decoded into vehicle routes by a capacity-first
greedy split (walk the tour, start a new route whenever adding the next customer
would exceed vehicle_capacity). This is the standard permutation encoding for
EA-based CVRP (Prins-style split) and lets every algorithm here -- QIEA, NSGA-II,
SPEA2, MOEA/D, RVEA -- operate on the exact same search space.

Objectives (all minimized):
  f1 distance   km,  sum of edge distances over all routes (return to depot each time)
  f2 time       h,   sum of edge travel times (road_factor makes this independent of f1)
  f3 cost       currency units, FIXED_COST_PER_VEHICLE * num_routes + variable fuel cost
                (depends on fleet size used, not just total distance)
  f4 emissions  kg CO2, distance + idling(time) + payload-dependent components; payload
                grows monotonically along a route (collection, not delivery), so this is
                genuinely sequence-dependent
  f5 balance    std. dev. of per-route completion time across vehicles (workload balance)

Constraints: capacity is enforced by construction (the split never lets a route's demand
exceed vehicle_capacity), so every decoded solution is feasible by design -- no repair
step and no penalty terms are needed. This directly avoids the repair-induced diversity
collapse reported for the qubit chromosome in the first paper (route1_334 in particular).
"""
import json
from pathlib import Path

import numpy as np
from pymoo.core.problem import Problem


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def split_routes(tour, demand, capacity):
    routes = []
    current, load = [], 0
    for c in tour:
        d = demand[c]
        if current and load + d > capacity:
            routes.append(current)
            current, load = [], 0
        current.append(c)
        load += d
    if current:
        routes.append(current)
    return routes


class CVRPInstance:
    """Precomputed arrays + evaluation logic for one Sibiu (or CVRPLIB) instance."""

    def __init__(self, data):
        self.name = data["name"]
        self.depot = data["depot_index"]
        self.n = data["num_points"]
        self.demand = np.array(data["demand"], dtype=float)
        self.capacity = float(data["vehicle_capacity"])
        self.dist = np.array(data["distance_matrix_km"])
        self.time = np.array(data["time_matrix_h"])
        self.cost_var = np.array(data["cost_matrix_variable"])
        a = data["assumptions"]
        self.fixed_cost_per_vehicle = a["fixed_cost_per_vehicle"]
        self.emission_per_km = a["emission_per_km"]
        self.emission_per_hour_idle = a["emission_per_hour_idle"]
        self.emission_load_factor = a["emission_load_factor"]
        self.customers = [i for i in range(self.n) if i != self.depot]

    @classmethod
    def from_file(cls, path):
        return cls(load_instance(path))

    def decode(self, perm):
        """perm: permutation of self.customers (0-based positions into self.customers)."""
        tour = [self.customers[i] for i in perm]
        return split_routes(tour, self.demand, self.capacity)

    def evaluate_routes(self, routes):
        total_dist = total_time = total_cost = total_emit = 0.0
        route_times = []
        for route in routes:
            seq = [self.depot] + route + [self.depot]
            load = 0.0
            r_dist = r_time = r_cost = r_emit = 0.0
            for a, b in zip(seq[:-1], seq[1:]):
                d = self.dist[a, b]
                t = self.time[a, b]
                r_dist += d
                r_time += t
                r_cost += self.cost_var[a, b]
                if b != self.depot:
                    load += self.demand[b]
                r_emit += (
                    self.emission_per_km * d
                    + self.emission_per_hour_idle * t
                    + self.emission_load_factor * d * load
                )
            total_dist += r_dist
            total_time += r_time
            total_cost += r_cost
            total_emit += r_emit
            route_times.append(r_time)
        total_cost += self.fixed_cost_per_vehicle * len(routes)
        balance = float(np.std(route_times)) if len(route_times) > 1 else 0.0
        return np.array([total_dist, total_time, total_cost, total_emit, balance])

    def evaluate_perm(self, perm):
        return self.evaluate_routes(self.decode(perm))


class CVRPProblem(Problem):
    """pymoo-facing wrapper: permutation of customer positions -> 5 objectives."""

    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        n_customers = len(instance.customers)
        super().__init__(n_var=n_customers, n_obj=5, xl=0, xu=n_customers - 1, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([self.instance.evaluate_perm(perm) for perm in X])
        out["F"] = F


if __name__ == "__main__":
    inst = CVRPInstance.from_file(Path(__file__).resolve().parent.parent / "data" / "processed" / "route2_199.json")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(inst.customers))
    routes = inst.decode(perm)
    print(f"{inst.name}: {len(routes)} routes, objectives = {inst.evaluate_routes(routes)}")
