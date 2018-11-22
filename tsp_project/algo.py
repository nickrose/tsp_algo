""" some algorithms for solving TSP problems """
import numpy as np
from collections import defaultdict
from copy import deepcopy
import itertools
from tsp_project.plot_data import plot_mail_route
from time import time
import pandas as pd
import warnings

default_solver = 'greedy_NN'


def zero():
    return 0


def import_failed(name, url):
    return (f'{name} TSP solver selected, but import '
        'failed, please ensure concode TSPSolver is '
        'available and installed (see '
        f'{url} for an '
        f'installation): switching to "{default_solver}"')


class TSP(object):
    """
        A simple object for storing a TSP problem setup and running various
        algorithms to determine approximate best paths.

        Nick Roseveare, Nov 2018
    """
    base_algorithm_types = ['quick_adjust', 'greedy_NN', 'greedy_NN_recourse',
        'random', 'concorde']

    def __init__(self, stop_locations, dist_mat_all, dist_mat_edge_limited,
            solver_type=default_solver, recourse_revisits=3,
            fail_on_solver_import=True, debug=1):
        """
            intialize the TSP problem object

            Args:
                stop_locations (np.matrix: N x 2): matrix of all possible stop
                    locations (2 is for (x,y) coordiante pairs).
                dist_mat_all (np.matrix NxN): raw ecludean distance between
                    all possible nodes in 'stop_locations'.
                dist_mat_edge_limited (np.matrix NxN): a more limited set of
                    edges representing allowable "walkways" between nodes. Edges
                    representing non-allowable paths between nodes have np.inf
                    values.
                solver_type (str, default is 'greedy_NN'): specifies which
                    solver to use, allowable set is:
                    'quick_adjust', 'greedy_NN', 'greedy_NN_recourse', 'random'
                recourse_revisits (int, default is 3): the number of recourse
                    steps to revisit in solving for base cycle with
                    'greedy_NN_recourse' solver.
                fail_on_solver_import (bool): if a solver import fails, then
                    produce an error, otherwise default solver is selected.
                    Default is False.
                debug (int, default is 1): the verbosity level to use.
            Returns:
                TSP object

        """
        self.stops = []
        self.debug = debug
        self.fail_on_solver_import = fail_on_solver_import
        self.solver_type = None
        self.update_solver(solver_type)
        self.stop_locations = stop_locations
        self.xrange = stop_locations[:,
        0].max() - stop_locations[:, 0].min()
        self.dist_mat_all = dist_mat_all
        self.dist_mat_edge_limited = dist_mat_edge_limited
        self.recourse_revisits = recourse_revisits
        self.solutions = {}
        self.is_monte_carlo_test = False
        self.mc_stats = None

    def update_solver(self, solver_type=None):
        if solver_type is None:
            solver_type = self.solver_type
        else:
            assert solver_type in self.base_algorithm_types, (
                f'solver_type must be one of {self.base_algorithm_types}')
            self.solver_type = solver_type
        if self.debug > 2:
            print(f'TSP using solver: {solver_type}')

    def add_stops(self, stops):
        """
            Add a list of ints that represents the index in the stop_locations
            array provided that should be solved for a route.
        """
        # print('input', stops)
        if not(isinstance(stops, list)):
            stops = [stops]
        elif isinstance(stops, np.ndarray):
            stops = stops.tolist()
        # remove repeats
        stops = [s for s in stops if s not in self.stops]
        self.stops.extend(deepcopy(stops))
        for k in self.solutions:
            self.solutions[k]['is_valid_for_stops'] = False
            self.solutions[k]['added_points'] = stops
            # print(f'{k}: added_points', self.solutions[k]['added_points'])

    def update_param(self, param, value):
        """
            Update a parameter and reset the availability of the preivous
            solution as accurate for the given configuration and parameters.
        """
        assert param in self.__dict__, f"specific param '{param}' not found"
        self.__dict__[param] = value
        for k in self.solutions:
            self.solutions[k]['is_valid_for_stops'] = False

    def reset_route_points(self):
        self.solutions = {}
        self.stops = []

    def init_monte_carlo(self):
        self.is_monte_carlo_test = True
        self.mc_stats = pd.DataFrame({})
        self.num_mc_runs = 0

    def run_solvers_one_monte_carlo(self, is_update=False, solver_list=None,
            auto_add_mc_stats=False, raise_error_invalid=True):
        if solver_list is None:
            solver_list = self.base_algorithm_types
        if not(is_update):
            # if this is a new set of stops and not an update, then reset the
            # solution set
            self.solutions = {}
        for solver in solver_list:
            self.solve(solver_type=solver, auto_add_mc_stats=auto_add_mc_stats,
                raise_error_invalid=raise_error_invalid)
        self.num_mc_runs += 1

    def finish_monte_carlo(self):
        self.is_monte_carlo_test = False
        stats = self.mc_stats
        self.mc_stats = None
        nmc = self.num_mc_runs
        self.num_mc_runs = 0
        return nmc, stats

    def solve(self, solver_type=None, auto_add_mc_stats=False,
            raise_error_invalid=True):
        self.update_solver(solver_type)
        nstops = len(self.stops)
        assert nstops, "no stops to attempt to connect"
        assert (isinstance(self.dist_mat_all, np.ndarray) and
            isinstance(self.dist_mat_edge_limited, np.ndarray) and
            (self.dist_mat_all.shape[0] >= nstops) and
            (self.dist_mat_edge_limited.shape[0] >= nstops)), (
                "invalid distance matrix")
        noncompute_return = (None, None, None)
        qa_solution = None
        if ((self.solver_type in self.solutions) and
                self.solutions[self.solver_type]['is_valid_for_stops']):
            valid = expnd_valid = True
        else:
            valid = expnd_valid = False
            route = []
            points = np.asarray(self.stops)
            if self.debug:
                print(f'finding approximate best routes for {len(points)} '
                    f'stops via {self.solver_type}')
                if self.debug > 1:
                    print(points)
            runtime = points_skipped = 0
            if self.solver_type != 'quick_adjust':
                startt = time()
                if self.solver_type == 'concorde':
                    imp_fail_str = None
                    try:
                        from concorde.tsp import TSPSolver
                    except ImportError as ie:
                        imp_fail_str = import_failed(
                            'concorde', 'https://github.com/nickrose/pyconcorde')
                        if self.fail_on_solver_import:
                            raise ImportError(imp_fail_str)
                        else:
                            warnings.warn(imp_fail_str)
                            self.solver_type = default_solver
                            startt = time()
                    else:
                        if self.debug > 2:
                            print('alt solver module: import successful')
                        x, y = (
                            np.asarray(self.stop_locations[self.stops,
                                0].reshape(len(self.stops)).tolist()[0]),
                            np.asarray(self.stop_locations[self.stops,
                                1].reshape(len(self.stops)).tolist()[0]))
                        solver = TSPSolver.from_data(x, y, 'EUC_2D')
                        solution = solver.solve()
                        valid = solution.found_tour
                        if self.debug > 2:
                            print(f'[{self.solver_type}]: valid: '
                                f'{solution.found_tour}, optimal_value: '
                                f'{solution.optimal_value:.4f}, tour indexed '
                                f'stops: {solution.tour}')
                        if valid:
                            route = np.asarray(self.stops)[solution.tour]
                            if self.debug > 2:
                                print(f'   tour: {route}')
                            route = route.tolist()
                            route.append(points[0])
                            route = np.asarray(route)
                if self.solver_type == 'random':
                    route = random_traveling_salesman(points, self.dist_mat_all,
                        avg_edges=None, start=0, end=0, debug=max(0, self.debug - 2))
                elif self.solver_type == 'greedy_NN':
                    route = nn_greedy_traveling_salesman(points, self.dist_mat_all,
                        start=0, end=0, debug=max(0, self.debug - 2))
                elif self.solver_type == 'greedy_NN_recourse':
                    route = nn_greedy_recourse_traveling_salesman(points,
                        self.dist_mat_all, start=0, end=0,
                        debug=max(0, self.debug - 2),
                        revisit_ndecisions=self.recourse_revisits)
                runtime += (time() - startt)
                if self.debug > 1:
                    print(f'routes [{self.solver_type:7s}]', route)

                if not(valid) and len(route):
                    valid = check_valid_path(points, route, self.dist_mat_all,
                        solver=self.solver_type, raiseError=raise_error_invalid)

                try:
                    startt = time()
                    route_expand, subroutes = connect_route_points_via_dijkstra(
                        route, self.dist_mat_edge_limited, debug=self.debug)
                    runtime += (time() - startt)
                except RuntimeError as rte:
                    if self.debug:
                        print(str(rte))
                        print('skipping this solve')
                    points_skipped = (len(
                        self.solutions[self.solver_type]['added_points'])
                        if ((self.solver_type in self.solutions) and
                            ('added_points' in self.solutions[self.solver_type]))
                        else len(points))
            else:
                if len(self.solutions) == 0:
                    if self.is_monte_carlo_test:
                        return
                    else:
                        if self.debug > 1:
                            print('no previous solutions to update, try '
                                'another solver')
                        return noncompute_return
                if 'concorde' in self.solutions:
                    qa_solution = 'concorde'
                elif 'greedy_NN' in self.solutions:
                    qa_solution = 'greedy_NN'
                elif 'greedy_NN_recourse' in self.solutions:
                    qa_solution = 'greedy_NN_recourse'
                else:
                    # we just need a previous saved solution
                    qa_solution = list(self.solutions.keys())[0]
                # find the two nearest points and recalculate the Dijkstra
                # sub-paths
                startt = time()
                route, subroutes, route_expand, points_skipped, valid = \
                    quick_adjust_route(
                        self.solutions[qa_solution]['route'],
                        self.solutions[qa_solution]['subroutes'],
                        self.solutions[qa_solution]['added_points'],
                        self.dist_mat_all,
                        self.dist_mat_edge_limited, debug=self.debug)
                runtime += (time() - startt)
            if self.debug > 2:
                print(f'check expanded path for "{self.solver_type}" solver')
                print(f'points skipped in expansion of path: {points_skipped}')
            if not(points_skipped) and route_expand is not None:
                expnd_valid, revisit_frac = check_valid_path(
                    points, route_expand, self.dist_mat_edge_limited,
                    solver=self.solver_type, allow_between_stops=True,
                    raiseError=raise_error_invalid)
            else:
                revisit_frac = None

            if (valid and expnd_valid) or not(raise_error_invalid):
                if (qa_solution is not None and
                        'added_points' in self.solutions[qa_solution]):
                    ref_added_points = self.solutions[qa_solution]['added_points']
                else:
                    ref_added_points = None
                is_update = (((self.solver_type in self.solutions) and
                    ('added_points' in self.solutions[self.solver_type]) or
                        (qa_solution is not None and
                        'added_points' in self.solutions[qa_solution])))
                self.solutions[self.solver_type] = dict(
                    is_valid_for_stops=(valid and expnd_valid),
                    dist_basic=(total_distance(
                        route, self.dist_mat_all) if valid else None),
                    dist_expand=(total_distance(
                        route_expand, self.dist_mat_edge_limited)
                        if expnd_valid else None),
                    route=(route if valid else
                        (self.solutions[self.solver_type]['route']
                        if (is_update and qa_solution is None) else None)),
                    route_expand=(route_expand if expnd_valid else
                        (self.solutions[self.solver_type]['route_expand']
                        if (is_update and qa_solution is None) else None)),
                    subroutes=(subroutes if expnd_valid else
                        (self.solutions[self.solver_type]['subroutes']
                        if (is_update and qa_solution is None) else None)),
                    runtime=runtime,
                    points_skipped=points_skipped,
                    is_update=is_update,
                    revisit_frac=revisit_frac,
                    recourse_revisits=self.recourse_revisits)

                if self.is_monte_carlo_test and auto_add_mc_stats:
                    self.mc_stats = self.all_route_stats(
                        solver_type=self.solver_type, stats=self.mc_stats)

                if ref_added_points:
                    self.solutions[self.solver_type][
                        'qa_previous_solve'] = qa_solution
                    self.solutions[self.solver_type].update(dict(
                        ref_added_points=ref_added_points))
        if not(self.is_monte_carlo_test):
            if self.debug:
                dist_basic = self.solutions[self.solver_type]['dist_basic']
                dist_expand = self.solutions[self.solver_type]['dist_expand']
                print(f"{self.solver_type} algorithm "
                    f"{('' if qa_solution is None else '(using prev solution: ' + qa_solution + ')')}")
                print(f"   runtime     : {self.solutions[self.solver_type]['runtime']:.3f}s")
                print(f"   route dist  : [coarse: {dist_basic:.3f}, fine: {dist_expand:.3f}]")
                print(f"   validity    : [{valid}] expanded[{expnd_valid}]")
                print(f"   revisit_frac: [{self.solutions[self.solver_type]['revisit_frac']:.3f}]")
            if valid and expnd_valid:
                return (self.solutions[self.solver_type]['route'],
                    self.solutions[self.solver_type]['route_expand'],
                    self.solutions[self.solver_type]['revisit_frac'])
            else:
                return noncompute_return

    def all_route_stats(self, solver_type=None,
            error_if_solver_type_na=True,
            stats=pd.DataFrame({})):
        """ collect some statistics on the solver runs """
        if len(self.solutions):
            solution_list = list(self.solutions.keys())
            if solver_type is not None:
                has_solver = (solver_type in self.solutions)
                if has_solver:
                    solution_list = [solver_type]
                elif error_if_solver_type_na:
                    raise KeyError(f'solver {solver_type} not found in '
                        'solutions')
            for solver in solution_list:
                data = dict(
                    solver=[solver],
                    runtime_s=[self.solutions[solver]["runtime"]],
                    revisit=[self.solutions[solver]["revisit_frac"]],
                    total_dist=[self.solutions[solver]["dist_expand"]],
                    is_update=[self.solutions[solver]["is_update"]],
                    points_skipped=[self.solutions[solver]["points_skipped"]],
                    revisit_ndec=[(self.recourse_revisits
                        if solver == 'greedy_NN_recourse' else None)])
                stats = stats.append(pd.DataFrame(data))
        return stats

    def run_time(self, solver_type=None):
        """ return the runtime for the specified algorithm """
        if len(self.solutions) and (solver_type in self.solutions):
            return self.solutions[solver_type]["runtime"]
        return None

    def plot_solutions(self, solver_type=None, title=None,
            plot_node_labels='stops_only', figsize=(8, 7)):
        if solver_type is None:
            solver_type = self.solver_type
        if solver_type not in self.solutions:
            raise Exception(f'no solution for solver_type[{solver_type}],'
                ' first call .solve()')
        update_title = (title is None)
        if update_title:
            title = f'{solver_type} route selection '
        alt_route = self.solutions[solver_type]['route']
        alt_route_label = 'direct solve'
        if solver_type == 'quick_adjust':
            if update_title:
                title += ('\nadded stops: '
                    f'{self.solutions[solver_type]["ref_added_points"]} ')
            if 'qa_previous_solve' in self.solutions[solver_type]:
                prev = self.solutions[solver_type]['qa_previous_solve']
                title += f'previous solution: {prev}'
                alt_route = self.solutions[prev]['route_expand']
                alt_route_label = 'previous solution'
        elif solver_type == 'greedy_NN_recourse':
            title += ('[recourse_revisits='
                f'{self.solutions[solver_type]["recourse_revisits"]}]')
        if update_title:
            title += (f'\ntotal_dist['
                f'{self.solutions[solver_type]["dist_expand"]:.2f}] revisit '
                f'fraction[{self.solutions[solver_type]["revisit_frac"]:.3f}]')

        plot_mail_route(self.solutions[solver_type]['route_expand'],
            self.stop_locations, self.stops,
            self.dist_mat_edge_limited,
            self.xrange,
            alt_route=alt_route,
            alt_route_label=alt_route_label,
            plot_node_labels=plot_node_labels, title=title, figsize=figsize)


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


def check_valid_path(points, route, distmat, solver, raiseError=True,
        allow_between_stops=False, debug=0):
    """ verify that a route 1) has all the stops that is should (and no more
        in the case of a strict check), and 2) only uses allowable walkways

        Args:
            points,
            route,
            distmat,
            solver,
            raiseError=True,
            allow_between_stops=False

        Returns:
            valid (bool) whether path is valid for the edge matrix costs and
                the points to reach with the cycle.
            revisit_frac (float): if revsiting a node is allowed, this is the
                fraction of revisits / total nodes visited. A heuristic measure
                the redundancy of the path.
    """
    # try:
    if isinstance(points, np.ndarray):
        points = set(points.tolist())
    else:
        points = set(points)
    if isinstance(route, np.ndarray):
        route = route.tolist()
    if debug > 3:
        print('input points to check', points)
        print('input route to check', route)
        print('unique route stops', set(route))
        print("allow_between_stops", allow_between_stops)
    if allow_between_stops:
        difference = points.difference(set(route))
        stops_present = (len(difference) == 0)
    else:
        difference = points.symmetric_difference(set(route))
        stops_present = (len(difference) == 0) and len(route) - 1 == len(points)
    # except TypeError as te:
    #     print('points', points)
    #     print('route', route)
    path_allowed = np.asarray([distmat[st, end]
            for st, end in zip(route[:-1], route[1:])])
    valid_paths_between = (~np.isinf(path_allowed)).all()
    if raiseError:
        not_allowed_paths = ', '.join([f'[{route[pidx]}, {route[pidx+1]}]'
            for pidx, pa in enumerate(path_allowed) if (np.isinf(pa))])
        # print(not_allowed_paths)
        assert stops_present, (f'[{solver}]: missing'
            f'{("" if allow_between_stops else " or extra")} stops: '
            f'{difference}')
        assert valid_paths_between, (
            f'[{solver}]: one or more paths[len:{len(path_allowed)}] between '
            f'stops are not allowed: {not_allowed_paths}')
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

        Args:
            route (list): current basic route
            distmat (np.matrix): the matrix of distances between all stops in
                the field of interest.
            ssubroutes (list of lists): the Dijkstra paths between the nodes
                in the 'route'.
            added_points (list): list of points to add to the basic route.
            distmat (np.matrix): the matrix of distances between all stops in
                the field of interest.
            distmat (np.matrix): the matrix of distances between all stops in
                the field of interest. np.inf values are in edges that are not
                passable between the [row, col] node pair.
            debug=0

        Returns:
            route (np.array): ordered points optimized according to distmat.
            subroutes (list of lists)
            expanded_route (list): the lists of subroutes chained together.
            valid (True)
    """
    if debug > 2:
        print('using quick_adjust() solve:')
        print(f'    route       : {route}')
        print(f'    added_points: {added_points}')
        print(f'    subroutes   : {subroutes}')

    added_points = deepcopy(added_points)
    if isinstance(route, np.ndarray):
        unq = np.unique(route)
        route = route.tolist()
    else:
        unq = set(route)
        unq = np.asarray(list(unq))
    # end_point = subroutes[-1][-1]

    nearest = get_nearest(added_points, unq, dist_mat_all)
    if len(nearest) == 0 or subroutes is None:
        if debug:
            print('all points are already in the route')
        if subroutes is not None:
            expanded_route = list(itertools.chain.from_iterable(subroutes))
        else:
            expanded_route = None

        return np.asarray(route), subroutes, expanded_route, 0, True

    if debug:
        print('quick adjust: nearest', len(nearest), nearest)
    next_tuple = list(sorted(iter(nearest), key=lambda x: (x[2])))[0]
    if debug > 1:
        print('next point to add', next_tuple)
    points_skipped = 0
    while next_tuple is not None:
        point, nearest_curr_idx, dist = next_tuple
        dist_mat_all[point, nearest_curr_idx]

        insert_index = route.index(nearest_curr_idx)
        if ((insert_index > 0) and (insert_index < len(route) - 1) and
                ((dist_mat_all[point, route[insert_index - 1]] +
                dist_mat_all[point, route[insert_index]]) >
                    (dist_mat_all[point, route[insert_index]] +
                    dist_mat_all[point, route[insert_index + 1]]))):
            # attempting to choose between insert before vs after the
            # nearest node selected
            insert_index += 1

            if debug > 1:
                print(f'nearest insert index (incremented 1): {insert_index} '
                    f'(native nearest index: {nearest_curr_idx})')
        elif debug > 1:
            print(f'nearest insert index: {insert_index} '
                f'(native nearest index: {nearest_curr_idx})')
        replacing_last_segment = (insert_index >= len(subroutes) - 1)

        start, end = ((subroutes[insert_index][0]
            if (insert_index < len(subroutes) - 1)
            else subroutes[-1][0]), (
            subroutes[insert_index + 1][0]
            if (insert_index + 1 < len(subroutes) - 1)
            else subroutes[-1][-1]))
        try:
            subroute_replace_1 = dijkstra(start, point, dist_mat_limited,
               include_endpoint=False)
            subroute_replace_2 = dijkstra(point, end, dist_mat_limited,
               include_endpoint=replacing_last_segment)

            if debug > 2:
                next_subr = (f', next subroute: {subroutes[insert_index+1]}'
                    if insert_index+1 < len(subroutes) else "")
                print(f'replacing subroute: {subroutes[insert_index]}{next_subr}')
            if len(subroute_replace_1):
                subroutes[insert_index] = subroute_replace_1
            if len(subroute_replace_2):
                if len(subroute_replace_1) == 0:
                    subroutes[insert_index] = subroute_replace_2
                elif replacing_last_segment:
                    subroutes.append(subroute_replace_2)
                else:
                    subroutes.insert(insert_index + 1, subroute_replace_2)
            if debug > 2:
                print(f'   with subroutes: \n{subroutes[insert_index]}\n   and '
                    f'\n{subroutes[insert_index+1]}')

            route.insert(insert_index, point)
            if debug > 2:
                print('updated route:')
                print([f'[{i}]: {rt}' for i, rt in enumerate(route)])
            unq = set(unq.tolist())
            unq.add(point)
            unq = np.asarray(list(unq))
        except RuntimeError as rte:
            if debug > 1:
                print(str(rte))
                print('skipping adding of this route')
                points_skipped += 1
        added_points.remove(point)
        next_tuple = None
        if len(added_points):
            nearest = get_nearest(added_points, unq, dist_mat_all)
            if len(nearest):
                next_tuple = list(sorted(iter(nearest), key=lambda x: (x[2])))[0]
                if debug > 1:
                    print('next point to add', next_tuple)

    expanded_route = list(itertools.chain.from_iterable(subroutes))

    return np.asarray(route), subroutes, expanded_route, points_skipped, True


def random_traveling_salesman(points, distmat, avg_edges=None, start=None,
        max_perm_samples=2e3, end=None, debug=0):
    """
    Finds the shortest route to visit all the cities by bruteforce.
    Time complexity is O(N!), so never use on long lists.

    We use a limit of max_perm_samples (default=2k) random samples of the
    permutation space of all possible routes and the select the route with
    the minimal overall route distance.

    Args:
        points,
        distmat (np.matrix): the matrix of distances between all stops in
            the field of interest.
        start=None,
        max_perm_samples=2e3,
        end=None,
        debug=0

    Returns:
        path (np.array): ordered points optimized according to distmat
    """
    if start is None:
        start = points[0]

    npoints = len(points)
    if avg_edges is None:
        nnodes = distmat.shape[0]
        nedges = sum([(~np.isinf(distmat[k, k+1:])).sum() for k in range(nnodes)])
        avg_edges = int(nedges/nnodes) + 1
    # attempt to estimate the number of possible routes given the average
    # number of edges per node
    nroutes_test = min(int(max_perm_samples), avg_edges**npoints)
    if debug:
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

    Args:
        points,
        distmat (np.matrix): the matrix of distances between all stops in
            the field of interest.
        start=None,
        end=None,
        strict=True whether the returned route has to be a Hamiltonian cycle
            (every stop occurs only once).
        debug=0

    Returns:
        path (np.array): ordered points optimized according to distmat
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
            if debug > 1:
                print('path not required to be strict Hamiltonian, returning'
                    f'to previous stop: {path[-2]}')
            path.append(path[-2])
        else:
            nearest = mv_vec[np.argmin(distmat[path[-1], mv_vec])]
            if debug > 1:
                print('distance to next selected stop: '
                    f'{distmat[path[-1], nearest]:.2f}')
    #         min(must_visit, key=lambda x: dist_mat[path[-1], x])
            path.append(nearest)
            must_visit.remove(nearest)
        if len(must_visit) == 0:
            break
    if end is not None:
        path.append(end)
    return np.asarray(path)


def nn_greedy_recourse_traveling_salesman(points, distmat, start=None, end=None,
        revisit_ndecisions=3, strict=True, debug=0):
    """
    As solving the problem in the brute force way is too slow,
    this function implements a simple heuristic: always
    go to the nearest city.

    In this version, however, we allow the most recent selection to be
    'swapped' with the

    Even if this algorithm is extremely simple, it works pretty well
    giving a solution only about 25% longer than the optimal one
    (cit. Wikipedia), and runs very fast in O(N^2) time complexity.

    Args:
        points,
        distmat,
        start=None,
        end=None,
        revisit_ndecisions=3,
        strict=True,
        debug=0
    Returns:
        path (np.array)
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
            if debug > 1:
                print('path not required to be strict Hamiltonian, returning'
                    f'to previous stop: {path[-2]}')
            path.append(path[-2])
        else:
            nearest = mv_vec[np.argmin(distmat[path[-1], mv_vec])]
            if debug > 1:
                print('distance to next selected stop: '
                    f'{distmat[path[-1], nearest]:.2f}')
    #         min(must_visit, key=lambda x: dist_mat[path[-1], x])
            path.append(nearest)
            must_visit.remove(nearest)

        # apply recourse to previous decisions
        revisit_dec = min(len(path) - 2, revisit_ndecisions)
        if revisit_dec > 0:
            num_dec_back_to_revisit = 1
            recent_path = deepcopy(path[
                -(revisit_dec + 1 + num_dec_back_to_revisit):-1])
            alter_path = None
            best_distance = init_best_dist = total_distance(recent_path, distmat)
            if debug > 1:
                recourse_update_str = ('shortest path route: best '
                    f'distance intial route: {best_distance:.2f}')
            current_node_pos = final_pos = len(recent_path) - 1
            recent_node = recent_path[current_node_pos]
            del recent_path[current_node_pos]
            for current_node_pos in range(revisit_dec):
                # swap the node positions
                recent_path.insert(current_node_pos, recent_node)
                ck_distance = total_distance(recent_path, distmat)
                if ck_distance < best_distance:
                    best_distance = ck_distance
                    if debug > 1:
                        recourse_update_str = (f'updated route: moved '
                            f'[{num_dec_back_to_revisit}] '
                            f'node (from head) to position '
                            f'[{current_node_pos-final_pos}] '
                            f'from head, best distance: {best_distance:.2f} < '
                            f'{init_best_dist:.2f}')
                    alter_path = deepcopy(recent_path)
                del recent_path[current_node_pos]
            if debug > 1:
                print(recourse_update_str)
            if alter_path is not None:
                ind = len(path) - (1 + num_dec_back_to_revisit)
                for new_order_point in reversed(alter_path):
                    path[ind] = new_order_point
                    ind -= 1
        # continue if more stops to visit
        if len(must_visit) == 0:
            break
    # delete consecutive repeats
    loc = 0
    while loc < len(path):
        if loc < len(path) - 1 and path[loc] == path[loc + 1]:
            del path[loc]
        else:
            loc += 1

    if end is not None:
        path.append(end)
    return np.asarray(path)


def connect_route_points_via_dijkstra(route, distmat, debug=1):
    """get the required list of intermediate points between the stops"""

    npoints_path = len(route)
    subroutes = [dijkstra(route[i], route[i+1], distmat,
       include_endpoint=False) for i in range(npoints_path - 1)]
    # include_endpoint=(i + 2 == npoints_path)) for i in range(npoints_path - 1)]
    subroutes.append([route[-1]])
    expanded_route = list(itertools.chain.from_iterable(
        [sr for sr in subroutes if len(sr)]))
    if debug > 1:
        print('expanded routes ', expanded_route)
    return expanded_route, subroutes


def dijkstra(start, end, distmat, include_endpoint=True):
    """ Implement shortest path search via Dijkstra

        Args:
            start (int)
            end (int)
            distmat (np.matrix),
            include_endpoint=True
        Returns:
            path (list)
    """
    nstops_all = distmat.shape[0]
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()
    all_stop_ind = np.arange(nstops_all)

    while current_node != end:
        visited.add(current_node)
        destinations = all_stop_ind[~np.isinf(distmat[current_node, :])]
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

        if not(len(next_destinations)):
            raise RuntimeError("Route Not Possible, current "
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
