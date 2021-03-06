# TSP ALGO
Some basic functionality for generating, plotting, and solving traveling
salesman route-planning problems.

Implements a basic greedy nearest neighbors approach
(https://en.wikipedia.org/wiki/Nearest_neighbor_search) as well as using
Dijkstra (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
for intermediate route finding.

To get started, just have a look at `interesting_paths.ipynb` in Jupyter 
notebook and execute the cells. The `tsp_solver.ipynb` has some examples
of how to use the self-contained solver class `TSP` (in `tsp_project/algo.py`).

As far as setup, first make sure to run `python setup.py develop` in the
from the `tsp_algo` folder, this should attempt to install all required
packages. Then try running the Jupyter notebook.
