""" some tools for plotting TSP solutions and graph data """
import numpy as np
# import tqdm
from matplotlib import pyplot as plt


def plot_mail_route(mail_route, stop_locations, mail_stop_indices,
        walking_routes, xrange, route_label='final route', route_color='r', po_index=0,
        alt_route=None, alt_route_label='direct solve', alt_route_color='b',
        islanded=None, plot_all_walkways=True, title=None,
        plot_node_labels='stops_only', figsize=(9, 8)):
    """ plot some of the graph information and overlay the route """

    plt.figure(figsize=figsize)
    nstops_all = stop_locations.shape[0]
    if plot_all_walkways:
        for k in range(nstops_all):
            for i in np.arange(k+1, nstops_all)[~np.isinf(walking_routes[k+1:, k])]:
                plt.plot([stop_locations[k, 0], stop_locations[i, 0]],
                         [stop_locations[k, 1], stop_locations[i, 1]], ":",
                         alpha=0.5,
                         linewidth=2.5, color='g')
    if (isinstance(plot_node_labels, str) and
            plot_node_labels == 'stops_only'):
        for k in mail_stop_indices:
            plt.text(stop_locations[k, 0]+xrange * 1e-3,
                     stop_locations[k, 1]+xrange * 1e-4, str(k),
                     fontdict=dict(size=10))
    elif plot_node_labels and not(isinstance(plot_node_labels, str)):
        for k in range(nstops_all):
            plt.text(stop_locations[k, 0]+xrange * 1e-3,
                     stop_locations[k, 1]+xrange * 1e-4, str(k),
                     fontdict=dict(size=7))

    # # plot all stops
    # plt.scatter(stop_locations[:, 0].tolist(), stop_locations[:, 1].tolist(),
    #     color='b', s=10)

    # plot mail stops
    plt.scatter(stop_locations[mail_stop_indices, 0].tolist(),
                stop_locations[mail_stop_indices, 1].tolist(), marker='X',
                color=route_color, s=45)

    # post office location
    plt.scatter([stop_locations[po_index, 0]], [stop_locations[po_index, 1]],
        marker='^', color='m', s=35)
    plt.text(stop_locations[po_index, 0]+xrange * 1e-3,
             stop_locations[po_index, 1]+xrange * 1e-4, 'start/end',
             fontdict=dict(size=10))

    legend = []
    # plot the route
    lg, = plt.plot(stop_locations[mail_route, 0],
            stop_locations[mail_route, 1], "--", linewidth=2.0, alpha=0.7,
            color=route_color, label=route_label)
    legend.append(lg)
    if alt_route is not None:
        lg, = plt.plot(stop_locations[alt_route, 0],
                stop_locations[alt_route, 1], "--", linewidth=1.5, alpha=0.7,
                color=alt_route_color, label=alt_route_label)
        legend.append(lg)

    if islanded is not None and len(islanded):
        lg = plt.scatter(
            stop_locations[islanded, 0].tolist(),
            stop_locations[islanded, 1].tolist(),
            color='y', s=18, label='possible islanded')
        legend.append(lg)
    if title is not None:
        plt.title(title, fontdict=dict(size=12))
    plt.ylim((stop_locations[:, 1].min(), stop_locations[:, 1].max() * 1.15))
    plt.xlabel('distance East-West (km)')
    plt.ylabel('distance South-North (km)')
    plt.legend(handles=legend)
    plt.show()
