import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

from networkx.utils.decorators import not_implemented_for, py_random_state

__all__ = ["randomized_partitioning", "mod_one_exchange", "mod_one_exchange_approximation"]

@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(1)
def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    """Compute a random partitioning of the graph nodes and its cut value.

    A partitioning is calculated by observing each node
    and deciding to add it to the partition with probability `p`,
    returning a random cut and its corresponding value (the
    sum of weights of edges connecting different partitions).

    Parameters
    ----------
    G : NetworkX graph

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    p : scalar
        Probability for each node to be part of the first partition.
        Should be in [0,1]

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_size : scalar
        Value of the minimum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a minimum cut.


    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    cut = {node for node in G.nodes() if seed.random() < p}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return cut_size, partition




def _swap_node_partition(cut, node):
    return cut - {node} if node in cut else cut.union({node})


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(2)
def mod_one_exchange(G, initial_cut=None, seed=None, weight=None):
    """Modified version of one exchange algorithm for computing a maximum cut.
    Instead of repeating the process till no improvement can be made, it stops after
    no improvement can be made and partitions are roughly equal in size."""

    """Compute a partitioning of the graphs nodes and the corresponding cut value.

    Use a greedy one exchange strategy to find a locally maximal cut
    and its value, it works by finding the best node (one that gives
    the highest gain to the cut value) to add to the current cut
    and repeats this process until no improvement can be made.

    Parameters
    ----------
    G : networkx Graph
        Graph to find a maximum cut for.

    initial_cut : set
        Cut to use as a starting point. If not supplied the algorithm
        starts with an empty cut.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_value : scalar
        Value of the maximum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a maximum cut.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    num_iters_max = G.number_of_nodes()*G.number_of_nodes()
    iterations = 0
    while True and num_iters_max > 0:
    # while True:
        # iterations += 1
        num_iters_max -= 1
        nodes = list(G.nodes())
        # Shuffling the nodes ensures random tie-breaks in the following call to max
        seed.shuffle(nodes)
        best_node_to_swap = max(
            nodes,
            key=lambda v: nx.algorithms.cut_size(
                G, _swap_node_partition(cut, v), weight=weight
            ),
            default=None,
        )
        prev_min_part_size = min(len(cut), len(G.nodes) - len(cut))
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)

        potential_min_part_size = min(len(potential_cut), len(G.nodes) - len(potential_cut))
        threshold = 1
        
        if potential_cut_size > current_cut_size :
            cut = potential_cut
            current_cut_size = potential_cut_size
        elif ((current_cut_size-potential_cut_size)<threshold and potential_min_part_size - prev_min_part_size > 0):
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break
    

    partition = (cut, G.nodes - cut)
    return current_cut_size, partition

def mod_one_exchange_approximation(G):
    n = G.number_of_nodes()
    partition_set = []

    # Calculate the best cut using the one exchange heuristic
    cut_size, partition_new = mod_one_exchange(G, partition_set)

    # Update the colors of the nodes based on the partition
    colors = [0] * n
    for i in partition_new[0]:
        colors[i] = 1

    return colors, cut_size

# Rewrite using classes

# class RandomizedPartitioning:
#     def __init__(self, G, seed=None, p=0.5, weight=None):
#         self.G = G
#         self.seed = seed
#         self.p = p
#         self.weight = weight

#     def randomized_partitioning(self):
#         cut = {node for node in self.G.nodes() if self.seed.random() < self.p}
#         cut_size = nx.algorithms.cut_size(self.G, cut, weight=self.weight)
#         partition = (cut, self.G.nodes - cut)
#         return cut_size, partition
    
# class ModOneExchange:
#     def __init__(self, G, initial_cut=None, seed=None, weight=None):
#         self.G = G
#         self.initial_cut = initial_cut
#         self.seed = seed
#         self.weight = weight

#     def _swap_node_partition(self, cut, node):
#         return cut - {node} if node in cut else cut.union({node})

#     def mod_one_exchange(self):
#         if self.initial_cut is None:
#             self.initial_cut = set()
#         cut = set(self.initial_cut)
#         current_cut_size = nx.algorithms.cut_size(self.G, cut, weight=self.weight)
#         num_iters_max = self.G.number_of_nodes()
#         iterations = 0
#         while True and num_iters_max > 0:
#             print(iterations)
#             iterations += 1
#             num_iters_max -= 1
#             nodes = list(self.G.nodes())
#             # Shuffling the nodes ensures random tie-breaks in the following call to max
#             self.seed.shuffle(nodes)
#             best_node_to_swap = max(
#                 nodes,
#                 key=lambda v: nx.algorithms.cut_size(
#                     self.G, self._swap_node_partition(cut, v), weight=self.weight
#                 ),
#                 default=None,
#             )
#             potential_cut = self._swap_node_partition(cut, best_node_to_swap)
#             potential_cut_size = nx.algorithms.cut_size(self.G, potential_cut, weight=self.weight)

#             partition1size = len(potential_cut)
#             partition2size = len(self.G.nodes) - partition1size
#             threshold = 1
            
#             if potential_cut_size > current_cut_size :
#                 cut = potential_cut
#                 current_cut_size = potential_cut_size
#             elif ((current_cut_size-potential_cut_size)<threshold and abs(partition1size - partition2size) > 1):
#                 cut = potential_cut
#                 current_cut_size = potential_cut_size
#             else:
#                 break
        
#         partition = (cut, self.G.nodes - cut)
#         return current_cut_size, partition

#     def mod_one_exchange_approximation(self):
#         n = self.G.number_of_nodes()
#         partition_set = []

#         # Calculate the best cut using the one exchange heuristic
#         cut_size, partition_new = self.mod_one_exchange()

#         # Update the colors of the nodes based on the partition
#         colors = [0] * n
#         for i in partition_new[0]:
#             colors[i] = 1

#         return colors, cut_size
    