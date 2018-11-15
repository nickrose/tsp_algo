""" some algorithms for solving TSP problems """
import numpy as np
from collections import defaultdict
from copy import deepcopy
import itertools
from tsp_project.plot_data import plot_mail_route


def zero():
    return 0


class TSP(object):

    def __init__(self, stops, dist_mat_all, dist_mat_edge_limited,
            solver_type='greedy_NN', debug=1):
        assert solver_type in ['greedy_NN', 'random', 'quick_adjust']
        self.stops = stops
        self.debug = debug
        self.solver_type = solver_type
        self.dist_mat_all = dist_mat_all
        self.dist_mat_edge_limited = dist_mat_edge_limited
        self.solutions = {}

    def add_stops(self, stops):
        if not(isinstance(stops, list)):
            stops = [stops]
        self.stops.extend(stops)
        for k in self.solutions:
            self.solutions[k]['is_valid_for_stops'] = False
            self.solutions[k]['added_points'] = stops

    def reset_route_points(self):
        self.solutions = {}
        self.stops = []

    def solve(self):
        nstops = len(self.stops)
        assert nstops, "no stops to attempt to connect"
        assert (isinstance(self.dist_mat_all) and isinstance(self.dist_mat_edge_limited)
            (self.dist_mat_all.shape[0] > nstops) and
            (self.dist_mat_edge_limited.shape[0] > nstops)), "invalid distance matrix"
        if ((self.solver_type in self.solutions) and
                self.solutions[self.solver_type]['is_valid_for_stops']):
            valid = expnd_valid = True
        else:
            points = self.stops.copy()

            print(f'finding approximate best routes for {len(points)} stops')
            print(points)

            if self.solver_type != 'quick_adjust':
                if self.solver_type == 'random':
                    route = random_traveling_salesman(points, self.dist_mat_all,
                        start=0, end=0)
                elif self.solver_type == 'greedy_NN':
                    route = nn_greedy_traveling_salesman(points, self.dist_mat_all,
                        start=0, end=0)
                if self.debug:
                    print('routes [{self.solver_type:7s}]', route)

                valid = check_valid_path(points, route, self.dist_mat_all)

                route_expand, subroutes = connect_route_points_via_dijkstra(route,
                    self.dist_mat_edge_limited)

            else:
                if len(self.solutions) == 0:
                    print('no previous solutions to update, try another solver')
                if 'greedy_NN' in self.solutions:
                    solver = 'greedy_NN'
                else:
                    # we just need a previous saved solution
                    solver = list(self.solutions.keys())[0]
                # find the two nearest points and recalculate the Dijkstra
                # sub-paths
                route, route_expand, subroutes, valid = quick_adjust_route(
                    self.solutions[solver]['route'],
                    self.solutions[solver]['subroutes'],
                    self.solutions[solver]['added_points'],
                    self.dist_mat_all,
                    self.dist_mat_edge_limited)

            expnd_valid, revisit_frac = check_valid_path(
                points, route_expand, self.dist_mat_edge_limited,
                allow_between_stops=True)

            if valid and expnd_valid:
                self.solutions[self.solver_type] = dict(
                    is_valid_for_stops=True,
                    dist_basic=total_distance(
                        route, self.dist_mat_all),
                    dist_expand=total_distance(
                        route_expand, self.dist_mat_edge_limited),
                    route=route,
                    route_expand=route_expand,
                    subroutes=subroutes,
                    revisit_frac=revisit_frac,
                    ref_added_points=self.solutions[solver]['added_points'])
        if self.debug:
            dist_basic = self.solutions[self.solver_type]['dist_basic']
            dist_expand = self.solutions[self.solver_type]['dist_expand']
            print(f"{self.solver_type} algorithm: dist_init:  "
                f"{dist_basic:.3f} "
                "[completed route: "
                f"{dist_expand:.3f}] "
                f"valid[{valid}] valid_expd[{expnd_valid}] "
                f"revisit_frac[{self.solutions[self.solver_type]['revisit_frac']}]")
        if valid and expnd_valid:
            return (self.solutions[self.solver_type]['route'],
                self.solutions[self.solver_type]['route_expand'],
                self.solutions[self.solver_type]['revisit_frac'])
        else:
            return (None, None, None)

    def plot_solutions(self, stop_locations, xrange, solver_type=None,
            plot_node_labels=False):
        if solver_type is None:
            solver_type = self.solver_type
        title = f'{solver_type} route selection: '
        if solver_type == 'quick_adjust':
            title += ('\nadded stops: '
                f'{self.solutions[solver_type]["ref_added_points"]} ')
            # plot_node_labels = True
        title += ('\nrevisit fraction'
            f'[{self.solutions[solver_type]["revisit_frac"]:.3f}]')
        plot_mail_route(self.solutions[solver_type]['route_expand'],
            stop_locations, self.stops, self.dist_mat_edge_limited, xrange,
                alt_route=self.solutions[solver_type]['route'],
                plot_node_labels=plot_node_labels, title=title)


def total_distance(points, dist_m):
    """
    Returns the length of the path passing throught
    all the points in the given order.
    """
    return np.asarray([
        dist_m[point, points[index + 1]]
        for index, point in enumerate(points[:-1])]).sum()


def check_is_island(k, others2check, walkways, dist_mat, max_nbhrs,
        checked=[], follow=0):
    """ heuristic for determining if a node in the graph is a
        part of an islanded group of nodes
    """
    nstops_all = dist_mat.shape[0]
    if follow > 15:
        if ((k in checked) and len(others2check) < max_nbhrs//2 and
                (np.array(others2check) < max_nbhrs//3).all()):
            # might be an island
            return True
        # might not be an island
        return False
    isinf_vec = np.isinf(walkways[:, k])
    if (isinf_vec).all():
        toadd = np.argsort(dist_mat[k, :])[1:max(3, max_nbhrs//2)]
        walkways[toadd, k] = walkways[k, toadd] = dist_mat[k, toadd]
    elif (~isinf_vec).sum() < 3:
        checked_ = deepcopy(checked)
        checked_.append(k)
        tocheck = np.arange(nstops_all)[~isinf_vec]
        nhr_counts = [(~np.isinf(walkways[:, i])).sum() for i in tocheck]
        if all([check_is_island(ck_idx, nhr_counts,
                    walkways, dist_mat, max_nbhrs, checked=checked_,
                    follow=follow+1) for ck_idx in tocheck]):
            return True
    return False


def check_valid_path(points, route, distmat, raiseError=True,
        allow_between_stops=False):
    """ verify that a route 1) has all the stops that is should (and no more
        in the case of a strict check), and 2) only uses allowable walkways

        Args:
            points,
            route,
            distmat,
            raiseError=True,
            allow_between_stops=False
        Returns:

    """
    points = set(points)
    if allow_between_stops:
        stops_present = (len(points.intersection(set(route))) == len(points))
    else:
        stops_present = (len(points.symmetric_difference(set(route))) == 0)
    path_allowed = np.asarray([distmat[st, end]
            for st, end in zip(route[:-1], route[1:])])
    valid_paths_between = (~np.isinf(path_allowed)).all()
    if raiseError:
        assert stops_present, f'missing or extra stops: {points.symmetric_difference(set(route))}'
        assert valid_paths_between, (f'one or more paths between stops are not allowed: ' +
            ', '.join([f'{route[pidx]} to {route[pidx+1]}'
                for pidx, pa in enumerate(path_allowed) if not(pa)]))
    if allow_between_stops:
        dd_revisit = defaultdict(zero)
        for stop in route:
            dd_revisit[stop] += 1
        revisit_frac = sum([int(dd_revisit[stop] > 1)
            for stop in route]) / len(route)
        return (stops_present and valid_paths_between), revisit_frac
    else:
        return stops_present and valid_paths_between


def get_nearest(added_points, unq_route_points, dist_mat_all):
    nearest = []
    for new_point in added_points:
        if new_point in unq_route_points:
            # if we already have this point, the route should already contain it
            continue
        # greedily attempt to augment the path near the closest current stop
        nhbr_dist = dist_mat_all[new_point, unq_route_points]
        near_idx = np.argsort(nhbr_dist)[0]
        nearest.append((new_point, unq_route_points[near_idx],
            nhbr_dist[near_idx]))
    return nearest


def quick_adjust_route(route, subroutes, added_points, dist_mat_all,
        dist_mat_limited, debug=0):
    """ attempt to modify the existing route with a fast heuristic which
        only has to decide the best place to insert the new stop followed
        by and adjusting of the navigation paths between stops, but only
        for the newly inserted stops
    """
    # nstops_all = dist_mat_all.shape[0]
    added_points = deepcopy(added_points)
    unq = np.unique(route)
    route = route.tolist()
    end_point = subroutes[-1][-1]

    nearest = get_nearest(added_points, unq, dist_mat_all)
    next_tuple = list(sorted(iter(nearest), key=lambda x: (x[2])))[0]
    if debug:
        print('next point to add', next_tuple)
    while next_tuple is not None:
        point, nearest_curr_idx, dist = next_tuple
        dist_mat_all[point, nearest_curr_idx]

        insert_index = route.index(nearest_curr_idx)
        if not(insert_index == len(route) - 1) and (
            insert_index == 0 or
                ((dist_mat_all[point, route[insert_index - 1]] +
                dist_mat_all[point, route[insert_index]]) >
                    (dist_mat_all[point, route[insert_index]] +
                    dist_mat_all[point, route[insert_index + 1]]))):
            # attempting to choose between insert before vs after the
            # nearest node selected
            insert_index += 1
        replacing_last_segment = (insert_index == len(route) - 1)

        start, end = route[insert_index], (
            end_point if replacing_last_segment else route[insert_index + 1])
        subroute_replace_1 = dijkstra(start, point, dist_mat_limited,
           include_endpoint=False)
        subroute_replace_2 = dijkstra(point, end, dist_mat_limited,
           include_endpoint=replacing_last_segment)

        route.insert(insert_index, point)
        subroutes[insert_index] = subroute_replace_1
        if replacing_last_segment:
            subroutes.append(subroute_replace_2)
        else:
            subroutes.insert(insert_index + 1, subroute_replace_2)

        added_points.remove(point)
        unq = set(unq.tolist())
        unq.add(point)
        unq = np.asarray(list(unq))
        if len(added_points):
            nearest = get_nearest(added_points, unq, dist_mat_all)
            next_tuple = list(sorted(iter(nearest), key=lambda x: (x[2])))[0]
            if debug:
                print('next point to add', next_tuple)
        else:
            next_tuple = None

    expanded_route = list(itertools.chain.from_iterable(subroutes))

    return route, subroutes, expanded_route, True


def random_traveling_salesman(points, distmat, start=None, end=None, debug=0):
    """
    Finds the shortest route to visit all the cities by bruteforce.
    Time complexity is O(N!), so never use on long lists.
    """
    if start is None:
        start = points[0]

    npoints = len(points)
    nnodes = distmat.shape[0]
    nedges = sum([(~np.isinf(distmat[k, k+1:])).sum() for k in range(nnodes)])
    avg_edges = int(nedges/nnodes) + 1
    nroutes_test = min(int(10e3), avg_edges**npoints)
    print(f'drawing {nroutes_test} random routes to test')
    # construct a limited set of random permutations
    if not(isinstance(points, np.ndarray)):
        points = np.asarray(points)
    else:
        points = points.copy()
    this_perm = points
    # permutes = []
    best_permute = None
    nvalid_found = 0
    best = np.inf
    while nvalid_found < nroutes_test:  # len(best_permute) < nroutes_test:
        np.random.shuffle(this_perm)
        if this_perm[0] == start:
            nvalid_found += 1
            # permutes.append(this_perm.copy())
            length = total_distance(this_perm, distmat)
            if length < best:
                best = length
                best_permute = this_perm.copy()
    # total_dist = np.zeros(len(permutes))
    # if debug:
    #     print(total_dist)
    # for pidx, perm in enumerate(permutes):
    #     total_dist[pidx] = total_distance(perm, distmat)
    # path = permutes[np.argsort(total_dist)[0]]

    path = best_permute
    if end is not None:
        path = path.tolist()
        path.append(end)
        return np.asarray(path)
    else:
        return path


def nn_greedy_traveling_salesman(points, distmat, start=None, end=None,
        strict=True, debug=0):
    """
    As solving the problem in the brute force way is too slow,
    this function implements a simple heuristic: always
    go to the nearest city.

    Even if this algorithm is extremely simple, it works pretty well
    giving a solution only about 25% longer than the optimal one
    (cit. Wikipedia), and runs very fast in O(N^2) time complexity.
    """
    if start is None:
        start = points[0]
    if isinstance(points, np.ndarray):
        must_visit = points.tolist()
    else:
        must_visit = deepcopy(points)
    path = [start]
    must_visit.remove(start)
    nstops = len(must_visit)
    for i in range(nstops):
        mv_vec = np.asarray(must_visit)
        if not(strict) and (np.isinf(distmat[path[-1], mv_vec])).all():
            # since we are not seeking a strict Hamiltonian cycle, allow
            # the path to revisit a previous stop
            assert len(path) > 1
            path.append(path[-2])
        else:
            nearest = mv_vec[np.argmin(distmat[path[-1], mv_vec])]
            if debug:
                print(f'distance to next selected stop: {distmat[path[-1], nearest]:.2f}')
    #         min(must_visit, key=lambda x: dist_mat[path[-1], x])
            path.append(nearest)
            must_visit.remove(nearest)
        if len(must_visit) == 0:
            break
    if end is not None:
        path.append(end)
    return np.asarray(path)


def connect_route_points_via_dijkstra(route, distmat, debug=1):
    """get the required list of intermediate points between the stops"""

    npoints_path = len(route)
    subroutes = [dijkstra(route[i], route[i+1], distmat,
       include_endpoint=(i + 2 == npoints_path)) for i in range(npoints_path - 1)]
    expanded_route = list(itertools.chain.from_iterable(subroutes))
    if debug > 1:
        print('expanded routes ', expanded_route)
    return expanded_route, subroutes


def dijkstra(start, end, distmat, include_endpoint=True):
    """ Implement shortest path search via Dijkstra"""
    nstops_all = distmat.shape[0]
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = np.arange(nstops_all)[~np.isinf(distmat[current_node, :])]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = distmat[current_node, next_node] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                best_weight = shortest_paths[next_node][1]
                if best_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node]
                             for node in shortest_paths
                             if node not in visited}

        assert len(next_destinations), ("Route Not Possible, current "
            f"node[{current_node}] no routes between {start} and {end}")

        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    if include_endpoint:
        return list(reversed(path))
    else:
        return list(reversed(path))[:-1]
