# This code is based on Kiran Garimella (https://github.com/gvrkiran/controversy-detection) implementation
# I update it to work with newer versions of python

import networkx as nx
import numpy as np


def get_random_sub_list(items: np.ndarray, k: int) -> list:
    """
    Returns a random sub list of size `k` from a given list `items`.
    """
    return list(np.random.choice(items, k, replace=False))


def perform_random_walk(G: nx.Graph, starting_node, same_side_nodes, other_side_nodes):
    """
    Perform a random walk that start from a given `starting_node` and ends when it reaches a node from `same_side_nodes` or `other_side_nodes`.
    Note that both `same_side_nodes` and `other_side_nodes` are lists of nodes. and they are a subset of nodes in each perspective side.

    Returns:
        `True` if the walk end in the same side and `False` otherwise.
    """

    is_end_in_same = True
    keep_going = True

    # step_count = 0

    current_node = starting_node

    while keep_going:

        current_node = np.random.choice(list(G.neighbors(current_node)))

        # To speed up the process, we can use the adjacency matrix instead of the graph object
        # adj_matrix = nx.to_numpy_matrix(G)
        # neighbors = np.nonzero(adj_matrix[current_node])[0]
        # current_node = np.random.choice(neighbors)

        # step_count += 1 # If needed

        if current_node in same_side_nodes:
            is_end_in_same = True
            keep_going = False

        if current_node in other_side_nodes:
            is_end_in_same = False
            keep_going = False

    return is_end_in_same


def count_walks(G, starting_side_nodes: list, ending_side_nodes: list):
    count_walk_end_in_same_side = 0
    count_walk_end_in_other_side = 0

    for i in range(len(starting_side_nodes) - 1):

        node = starting_side_nodes[i]

        other_nodes = starting_side_nodes[:i] + starting_side_nodes[i + 1 :]

        is_same_side: bool = perform_random_walk(
            G, node, other_nodes, ending_side_nodes
        )

        if is_same_side:
            count_walk_end_in_same_side += 1
        else:
            count_walk_end_in_other_side += 1

    return count_walk_end_in_same_side, count_walk_end_in_other_side


def RWC(
    G: nx.Graph, left_nodes: list, right_nodes: list, itr_num=1000, percent: float = 0.1
) -> float:
    """
    Random Walk Controversy Metric

    Input:
        G: NetworkX graph
        left: list of users in the left side (1st community)
        right: list of users in the right side (2nd community)
        itr_num: number of iterations
        percent: percentage of users from each side to be used in each iteration

    Returns:
        A float value
    """

    # start_end
    left_left = 0
    left_right = 0
    right_right = 0
    right_left = 0

    left_percent = int(percent * len(left_nodes))
    right_percent = int(percent * len(right_nodes))

    for _ in range(itr_num):

        user_nodes_left = get_random_sub_list(left_nodes, left_percent)
        user_nodes_right = get_random_sub_list(right_nodes, right_percent)

        # Staring from the left
        end_in_left, end_in_right = count_walks(G, user_nodes_left, user_nodes_right)
        left_left += end_in_left  # Count the walks that sated in left and ended in left
        left_right += end_in_right

        # Staring from the right
        end_in_right, end_in_left = count_walks(G, user_nodes_right, user_nodes_left)
        right_right += end_in_right
        right_left += end_in_left

    e1 = left_left * 1.0 / (left_left + right_left)
    e2 = left_right * 1.0 / (left_right + right_right)
    e3 = right_left * 1.0 / (left_left + right_left)
    e4 = right_right * 1.0 / (left_right + right_right)

    return e1 * e4 - e2 * e3
