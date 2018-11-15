""" some tools for generating TSP problems and graph data """
# needed to add imports
import numpy as np
# import pickle
# import os
# from collections import defaultdict
# import tqdm
import scipy.sparse as sprs
from tsp_project.algo import check_is_island


def dist_metric(x, y):
    return np.linalg.norm(x - y)


def generate_graph(nstops_all=1000, base_dist=2., grid_size_noise_std=1.,
        seed=40, debug=1):
    """ generate data in which path problems can be formulated

        Args:
            nstops_all=1000,
            base_dist=2., base distance from origin (in km)
            grid_size_noise_std=1.,
            seed=40,
            debug=1
        Returns:
            stop_locations,
            dist_mat,
            xrange
    """
    np.random.seed(seed=seed)
    grid_range = np.matrix([[-base_dist, base_dist],
                            [-base_dist, base_dist]]
                            ) + grid_size_noise_std * np.random.uniform(size=(2, 2))
    xrange = np.diff(grid_range[0, :])[0, 0]
    if debug > 1:
        print('grid_range', grid_range)
    # intialize field of random stops
    stop_locations = np.concatenate([
        np.matrix(np.random.uniform(size=nstops_all) * np.diff(
            grid_range[0, :])[0, 0] + grid_range[0, 0]).transpose(),
        np.matrix(np.random.uniform(size=nstops_all) * np.diff(
            grid_range[1, :])[0, 0] + grid_range[1, 0]).transpose()], axis=1)
    stop_locations[0, :] = np.array([0, 0])  # the post office

    dist_mat = np.zeros((nstops_all, nstops_all))
    for i in range(nstops_all):
        for j in range(i + 1, nstops_all):
            dist_mat[i, j] = dist_metric(
                stop_locations[i, :],
                stop_locations[j, :])

    # we filled the upper triangular portion of the matrix, fill the
    # lower triangular part now
    dist_mat += dist_mat.transpose()

    print('final distance matrix shape', dist_mat.shape)
    return stop_locations, dist_mat, xrange


def generate_walkways_from_complete_graph(dist_mat, xrange, max_nbhrs=10,
        walkway_dist_laplace_param=0.12, fix_islands=False, seed=46, debug=1):
    walkways = dist_mat.copy()
    nstops_all = dist_mat.shape[0]
    dist_vec = []
    np.random.seed(seed=seed)
    for k in range(nstops_all):
        # use laplace distribution to get heavy tailed
        # distribution of possible walkways to nearby stops
        dist = np.abs(np.random.laplace(
            scale=walkway_dist_laplace_param, size=1)) * xrange
        dist_vec.append(dist)
        neighbor_dist = walkways[:, k]
        neighbor_dist[neighbor_dist > dist] = np.inf
        if (~np.isinf(neighbor_dist)).sum() > max_nbhrs:
            keep_nhbr = max_nbhrs
            rmv_nbhr_connections = np.argsort(neighbor_dist)[keep_nhbr:]
            keepdist = np.inf
            if max_nbhrs < min(neighbor_dist.shape[0], max_nbhrs * 2):
                keep_rand_outlier = np.random.randint(max_nbhrs, min(
                    neighbor_dist.shape[0], max_nbhrs * 2))
                keepdist = neighbor_dist[keep_rand_outlier]
            neighbor_dist[rmv_nbhr_connections] = np.inf
            if keep_rand_outlier < neighbor_dist.shape[0]:
                neighbor_dist[keep_rand_outlier] = keepdist
        neighbor_dist[k] = np.inf
        walkways[:, k] = walkways[k, :] = neighbor_dist

    dist_vec = np.asarray(dist_vec)
    if debug:
        print(f'number of init connecting paths: {(~np.isinf(walkways)).sum()} | '
              f'allowed dist {np.mean(dist_vec):.3f} +/- {np.std(dist_vec):.3f}')

    # make sure no points are isolated
    islanded = np.asarray([check_is_island(
            k, (~np.isinf(walkways[k, :])).sum(),
            walkways, dist_mat, max_nbhrs)
        for k in range(nstops_all)])
    if fix_islands:
        # fix islands
        for k in np.arange(nstops_all)[islanded]:
            isNinf = ~np.isinf(walkways[k, :])
            neighbors = dist_mat[k, :]
            neighbors[isNinf] = np.inf
            toadd = np.argsort(neighbors)[1:max(5, max_nbhrs//2)]
            walkways[k, toadd] = neighbors[toadd]
        # recheck
        islanded = np.asarray([check_is_island(
                k, (~np.isinf(walkways[k, :])).sum(),
                walkways, dist_mat, max_nbhrs)
            for k in range(nstops_all)])

    unconnected = sum(islanded)
    print(f'number of questionably connected stops: {unconnected}')
    islanded = np.arange(nstops_all)[islanded]
    if debug > 1:
        for k in islanded:
            ww = walkways[k, :].copy()
            ww[np.isinf(ww)] = 0
            print(f'   walkways[{k}, :] =\n {sprs.csr_matrix(ww)}')
    return walkways, islanded


def generate_mail_stops(dist_mat, xrange, nstops_mail=30, manual_add=[],
        mail_stop_dist_laplace_param=0.22, po_start=0, seed=47, debug=1):
    """ generate list of stops on a mail route to visit """
    np.random.seed(seed=seed)
    # mail_locations = np.zeros((nstops_mail, 2))
    # weight selection of mail stops away from each other
    mail_stop_indices = [po_start]
    for k in range(nstops_mail):
        # use laplace distribution to get heavy tailed
        # result of distance from last stop
        dist = np.abs(np.random.laplace(
            scale=mail_stop_dist_laplace_param, size=1)) * xrange
        poffice_weight = k/nstops_mail
        # weight the selection of a point to be nearby, at the end of the path
        # weight the selection toward returning to the post office
        nneighbors = np.argsort(
            np.abs(dist_mat[mail_stop_indices[-1], :] - dist) * (
                1-poffice_weight) +
            np.abs(dist_mat[0, :] - dist) * poffice_weight)
        for sel in nneighbors:
            if sel not in mail_stop_indices:
                break
        mail_stop_indices.append(sel)

    if len(manual_add):
        ms_vec = np.asarray(mail_stop_indices)
        for k in manual_add:
            mail_stop_indices.insert(
                mail_stop_indices.index(ms_vec[np.argmin(dist_mat[k, ms_vec])]),
                k)
    if debug:
        print('mail stops to visit')
        print(mail_stop_indices)
    return mail_stop_indices


def generate_interesting_stops(dist_mat, xrange, nstops_interesting=2,
        stop_dist_laplace_param=0.5, start_loc=0, seed=48, debug=1):
    """ generate list of 'interesting' stops to visit """
    np.random.seed(seed=seed)
    # mail_locations = np.zeros((nstops_mail, 2))
    # weight selection of mail stops away from each other
    stop_indices = []
    for k in range(nstops_interesting):
        # use laplace distribution to get heavy tailed
        # result of distance from last stop
        dist = np.abs(np.random.laplace(
            scale=stop_dist_laplace_param, size=1)) * xrange
        # weight the selection of a point to be nearby, at the end of the path
        # weight the selection toward returning to the post office
        nneighbors = np.argsort(
            np.abs(dist_mat[start_loc, :] - dist))
        for sel in nneighbors:
            if sel not in stop_indices:
                break
        stop_indices.append(sel)

    if debug:
        print('interesting stops to visit')
        print(stop_indices)
    return stop_indices
