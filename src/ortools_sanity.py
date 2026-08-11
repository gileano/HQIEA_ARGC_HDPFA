"""
OR-Tools exact/near-optimal sanity check for small CVRP instances: solves the
classic single-objective (minimize total distance) CVRP so we have a trusted
reference point to judge how far the Pareto-front distance dimension is from
a near-optimal solution. This is NOT multi-objective -- it is the "does our
distance objective make sense at all" check the plan calls for.
"""
import sys
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import CVRPInstance  # noqa: E402


def solve_cvrp_distance(instance: CVRPInstance, num_vehicles, time_limit_s=20):
    n = instance.n
    dist = instance.dist
    scale = 1000  # OR-Tools wants integer costs
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, instance.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a, b = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(dist[a, b] * scale)

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    demand = instance.demand.astype(int)

    def demand_callback(from_index):
        return int(demand[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(instance.capacity)] * num_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    total_distance = 0.0
    routes = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != instance.depot:
                route.append(node)
            prev = index
            index = solution.Value(routing.NextVar(index))
            total_distance += dist[manager.IndexToNode(prev), manager.IndexToNode(index)]
        if route:
            routes.append(route)
    return total_distance, routes


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "A-n32-k5.json"
    inst = CVRPInstance.from_file(path)
    # A-n32-k5 is named for 5 vehicles in the original CVRPLIB instance.
    result = solve_cvrp_distance(inst, num_vehicles=5, time_limit_s=15)
    if result is None:
        print("no feasible solution found")
    else:
        total_distance, routes = result
        print(f"{inst.name}: OR-Tools near-optimal distance = {total_distance:.2f} over {len(routes)} routes")
        full_obj = inst.evaluate_routes(routes)
        print("full 5-objective vector for this solution:", full_obj)
