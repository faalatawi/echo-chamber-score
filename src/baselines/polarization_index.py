import networkx as nx
from networkx import Graph
import numpy as np


def add_ideology_to_graph(
    G: Graph, community_label: np.ndarray, percentage: float = 0.05
) -> Graph:
    """
    Input:
        - G: A Graph of users and their connections
        - community_label: A vector with the community label of each node
        - percentage: The percentage of nodes to be considered elite

    Output:
        - G: The same graph with the ideology attribute added to the nodes
    """

    degrees = G.degree()
    degrees = dict(degrees)
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    num_0 = max(int(np.sum(community_label == 0) * percentage), 1)
    num_1 = max(int(np.sum(community_label == 1) * percentage), 1)

    top_0 = []
    top_1 = []

    for node, _ in sorted_degrees:
        if len(top_0) == num_0 and len(top_1) == num_1:
            break
        if community_label[node] == 0 and len(top_0) < num_0:
            top_0.append(node)
        elif community_label[node] == 1 and len(top_1) < num_1:
            top_1.append(node)

    # Add the ideology attribute to the nodes
    # if in top_0, then 1, if in top_1, then -1, else 0
    for node in G.nodes():
        if node in top_0:
            G.nodes[node]["ideology"] = 1
        elif node in top_1:
            G.nodes[node]["ideology"] = -1
        else:
            G.nodes[node]["ideology"] = 0

    core_nodes = top_0 + top_1

    return G, core_nodes


def opinion_model(G: Graph, core_nodes: list, tol=10**-5) -> np.ndarray:
    """
    Input:
    - G: Graph to calculate opinions. The nodes have an attribute "ideology" with their ideology, set to 0 for all listeners, 1 and -1 for the elite.
    - core_nodes: Nodes that belong to the seed (Identifiers from the Graph G)
    - tol: is the threshold for convergence. It will evaluate the difference between opinions at time t and t+1

    Output:
    - v_current: Vector with the opinions of the nodes
    """

    Aij = nx.to_numpy_array(G, weight=None)

    # Build the vectors with users opinions
    v_current = []
    for node_id in G.nodes():
        v_current.append(1.0 * G.nodes[node_id]["ideology"])
    v_current = np.array(v_current)

    not_converged = len(v_current)

    # Do as many times as required for convergence
    while not_converged > 0:
        v_new = np.zeros_like(v_current)

        for j in G.nodes():
            v_new[j] = np.mean(v_current[Aij[j] == 1])

        # keep the opinion of the core nodes
        for j in core_nodes:
            v_new[j] = v_current[j]

        diff = np.abs(v_current - v_new)
        not_converged = len(diff[diff > tol])

        v_current = v_new

    return v_current


def get_polarization_index(ideos: np.ndarray):
    # Input: Vector with individuals Xi
    # Output: Polarization index, Area Difference, Normalized Pole Distance
    D = []  # POLE DISTANCE
    AP = []  # POSSITIVE AREA
    AN = []  # NEGATIVE AREA
    cgp = []  # POSSITIVE GRAVITY CENTER
    cgn = []  # NEGATIVE GRAVITY CENTER

    ideos.sort()
    hist, edg = np.histogram(ideos, np.linspace(-1, 1.1, 50))
    edg = edg[: len(edg) - 1]
    AP = sum(hist[edg > 0])
    AN = sum(hist[edg < 0])
    AP0 = 1.0 * AP / (AP + AN)
    AN0 = 1.0 * AN / (AP + AN)
    cgp = sum(hist[edg > 0] * edg[edg > 0]) / sum(hist[edg > 0])
    cgn = sum(hist[edg < 0] * edg[edg < 0]) / sum(hist[edg < 0])
    D = cgp - cgn
    p1 = 0.5 * D * (1.0 - 1.0 * abs(AP0 - AN0))  # polarization index
    DA = 1.0 * abs(AP0 - AN0) / (AP0 + AN0)  # Areas Difference
    D = 0.5 * D  # Normalized Pole Distance
    return p1, DA, D


def my_implementation_polarization_index(X) -> float:
    A_negative = X[X <= 0.0]
    A_positive = X[X > 0.0]

    # Probability of A+
    P_A_plus = len(A_positive) / len(X)

    # Probability of A-
    P_A_minus = len(A_negative) / len(X)

    # The deference between the two probabilities
    Delta_A = np.abs(P_A_plus - P_A_minus)

    # Center of gravity of A+ and A-
    gc_neg = np.mean(A_negative)
    gc_pos = np.mean(A_positive)

    # The distance between the two centers of gravity
    d = np.abs(gc_pos - gc_neg) / np.abs(np.max(X) - np.min(X))

    # The polarization index
    u = (1.0 - Delta_A) * d

    return u
