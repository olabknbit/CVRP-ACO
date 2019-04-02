"""Capacited Vehicles Routing Problem (CVRP) using an Ant Colony Algorithm (ACO)."""

from __future__ import print_function

import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

ALPHA = 1
BETHA = 1
Q = 1
RHO = 0.1

nk = {5: [32, 33, 34, 36, 37, 38, 39], 6: [33, 37, 39, 45], 7: [45, 46, 48, 53, 54], 9: [55, 61, 65]}


def get_problem_sol_file_pair(n, k):
    problem_fn = 'data/A-VRP/A-n' + str(n) + '-k' + str(k) + '.vrp'
    sol_fn = 'data/A-VRP-sol/opt-A-n' + str(n) + '-k' + str(k)
    write_fn = 'data/A-VRP-my/latest-A-n' + str(n) + '-k' + str(k)
    return problem_fn, sol_fn, write_fn


class Coords:
    def __init__(self, filenames):
        problem_fn, sol_fn, self.write_fn = filenames
        self.capacities = []
        reading_coords = False
        reading_demand = False
        reading_depot = False
        self.no_trucks = 5
        self.coords = []
        self.demands = []
        self.depot = None
        with open(problem_fn, 'r') as f:
            for line in f.readlines():
                contents = line.split(' ')
                if 'CAPACITY' in contents[0]:
                    capacity = int(contents[2])
                    self.capacities = [capacity for _ in range(self.no_trucks)]
                if 'NODE_COORD_SECTION' in contents[0]:
                    reading_coords = True
                    continue
                if 'DEMAND_SECTION' in contents[0]:
                    reading_demand = True
                    reading_coords = False
                    continue
                if 'DEPOT_SECTION' in contents[0]:
                    reading_depot = True
                    reading_demand = False
                    reading_coords = False
                    continue

                if reading_coords:
                    self.coords.append([int(contents[1]), int(contents[2])])

                if reading_demand:
                    self.demands.append(int(contents[1]))

                if reading_depot:
                    self.depot = int(contents[1]) - 1
                    break
        self.distance_matrix = self._create_distance_matrix()

        with open(sol_fn, 'r') as f:
            for line in f.readlines():
                contents = line.split(' ')
                if 'cost' in contents[0].lower():
                    self.cost = int(contents[1])
                    break

    def _create_distance_matrix(self):
        distances_m = []
        for i_coord in self.coords:
            distances = []
            for j_coord in self.coords:
                # distance = np.linalg.norm(np.subtract(i_coord, j_coord))
                x1, y1 = i_coord
                x2, y2 = j_coord
                distance = int(np.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)))
                distances.append(distance)
            distances_m.append(distances)
        return distances_m


def create_data_model():
    """Stores the data for the problem."""
    coords = Coords()
    data = {}
    data['distance_matrix'] = coords.distance_matrix
    data['demands'] = coords.demands
    data['vehicle_capacities'] = coords.capacities
    data['num_vehicles'] = coords.no_trucks
    data['depot'] = coords.depot
    return data


def get_attractiveness_of_city(distance):
    return 1 / distance


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index), route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def solve_using_google():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


class Pheromone_Trails:
    def __init__(self, no_cities):
        from random import uniform
        self.pheromones_matrix = []
        for distances in range(no_cities):
            pheromones = [uniform(0, 0.1) for _ in range(no_cities)]
            self.pheromones_matrix.append(pheromones)

    def get_pheromone_trail(self, city_i, city_j):
        return self.pheromones_matrix[city_i][city_j]

    def evaporate(self):
        for i in range(len(self.pheromones_matrix)):
            for j in range(len(self.pheromones_matrix[0])):
                self.pheromones_matrix[i][j] *= (1 - RHO)

    def update(self, city_i, city_j, overall_trip_distance):
        delta_tau = Q / overall_trip_distance
        self.pheromones_matrix[city_i][city_j] *= delta_tau
        self.pheromones_matrix[city_j][city_i] *= delta_tau


class Ant:
    def __init__(self, capacity, depot, demands, distance_m, pheromone_trails):
        self.current_city = depot
        self.distance_m = distance_m
        self.not_visited = self.create_not_visited(distance_m, depot)
        self.capacity = capacity
        self.load = capacity
        self.demands = demands
        self.trips = []
        self.current_trip = []
        self.pheromone_trails = pheromone_trails
        self.trips_distance = None

        self.start()

    def create_not_visited(self, distance_m, depot):
        nv = set()
        for i, _ in enumerate(distance_m):
            if i != depot:
                nv.add(i)
        return nv

    def start(self):
        from random import randint
        start_city = randint(1, len(self.distance_m) - 1)
        self.visit_city(start_city)

    def visit_city(self, index):
        self.not_visited.remove(index)
        self.load -= self.demands[index]
        distance = self.distance_m[self.current_city][index]
        self.current_city = index
        self.current_trip.append((index, distance))

    def visit_depot(self):
        depot_index = 0
        self.load = self.capacity
        distance = self.distance_m[self.current_city][depot_index]
        self.current_city = depot_index
        self.current_trip.append((depot_index, distance))
        self.trips.append(self.current_trip)
        self.current_trip = []

    def calc_val(self, nv_i):
        pheromone_val = self.pheromone_trails.get_pheromone_trail(self.current_city, nv_i)
        attractiveness = get_attractiveness_of_city(self.distance_m[self.current_city][nv_i])
        if self.current_city == nv_i:
            print(self.current_city, nv_i)
            exit(1)
        return pow(pheromone_val, ALPHA) * pow(attractiveness, BETHA)

    def create_path(self):
        while len(self.not_visited) > 0:
            # TODO implement neighbours later
            neighbours = []
            if len(neighbours) == 0:
                nv = list(self.not_visited)
                nv_values = [self.calc_val(nv_i) for nv_i in nv]
                next_to_visit = nv[nv_values.index(max(nv_values))]

                if self.demands[next_to_visit] > self.load:
                    # Come back to the depot to reload
                    self.visit_depot()
                else:
                    self.visit_city(next_to_visit)
        self.visit_depot()

    def calculate_paths_quality(self):
        overall_distance = 0
        for trip in self.trips:
            for _, dist in trip:
                overall_distance += dist
        self.trips_distance = overall_distance

    def leave_pheromone(self):
        for trip in self.trips:
            prev_city = 0
            for i, (city, _) in enumerate(trip):
                if i == len(trip) - 1:
                    break
                self.pheromone_trails.update(prev_city, city, self.trips_distance)
                prev_city = city

    def reset(self):
        self.current_city = 0
        self.not_visited = self.create_not_visited(self.distance_m, 0)
        self.current_trip = []
        self.trips = []
        self.load = self.capacity
        self.start()


def trips_to_str(trips):
    s = ''
    for index, trip in enumerate(trips):
        line = 'Route #' + str(index + 1) + ": "
        for city, _ in trip:
            if city ==0:
                break
            line += str(city) + " "

        s += line + '\n'
    return s


def save(filename, optimal, my_best, routes):
    with open(filename, 'w')as f:
        f.write(routes)
        f.write("my_best: " + str(my_best) + '\n')
        f.write("optimal: " + str(optimal))


def solve_using_ants():
    no_ants = 100
    iterations = 15

    overall_score = 0

    for k, ns in nk.items():
        k_score = 0
        for n in ns:

            data = Coords(get_problem_sol_file_pair(n, k))
            no_cities = len(data.distance_matrix)
            pheromone_trails = Pheromone_Trails(no_cities)
            ants = [Ant(capacity=data.capacities[0], depot=data.depot, demands=data.demands,
                        distance_m=data.distance_matrix, pheromone_trails=pheromone_trails) for _ in range(no_ants)]
            import sys
            best_trip = sys.maxsize
            routes = ''
            for iteration in range(iterations):
                for ant in ants:
                    ant.create_path()
                    ant.calculate_paths_quality()
                    if ant.trips_distance < best_trip:
                        best_trip = ant.trips_distance
                        routes = trips_to_str(ant.trips)

                for ant in ants:
                    pheromone_trails.evaporate()
                    ant.leave_pheromone()
                    ant.reset()
            metric = (best_trip - data.cost) / data.cost
            print("n", n, "k", k, "optimal", data.cost, " actual", best_trip, 'metric', metric)
            save(data.write_fn, data.cost, best_trip, routes)
            # print(routes)
            # print('---------------------------------------')
            k_score += metric
        print('k', k, 'k_score', k_score)
        overall_score += k_score
    print('overall_score', overall_score)


def main():
    # solve_using_google()
    solve_using_ants()
    # TODO(olab) please write your own impl


if __name__ == '__main__':
    main()
