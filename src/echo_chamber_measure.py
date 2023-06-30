from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np


class EchoChamberMeasure:
    def __init__(
        self,
        users_representations: np.ndarray,
        labels: np.ndarray,
        metric: str = "euclidean",
    ):
        if metric == "euclidean":
            self.distances = euclidean_distances(users_representations)
        elif metric == "cosine":
            self.distances = cosine_distances(users_representations)
        self.labels = labels

    # `a` part of the metric
    def cohesion_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        node_distances = self.distances[idx, self.labels == node_label]

        return np.mean(node_distances)

    # `b` part of the metric
    def separation_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        dist = []
        for l in np.unique(self.labels):
            if l == node_label:
                continue
            dist.append(np.mean(self.distances[idx, self.labels == l]))

        return np.min(dist)

    def metric(self, idx: int) -> float:
        a = self.cohesion_node(idx)
        b = self.separation_node(idx)

        return (-a + b + max(a, b)) / (2 * max(a, b))

    def echo_chamber_index(self) -> float:
        nodes_metric = []
        for i in range(self.distances.shape[0]):
            nodes_metric.append(self.metric(i))
        return np.mean(nodes_metric)

    def community_cohesion(self, community_label: int) -> float:
        com_coh = []

        for i in range(self.distances.shape[0]):
            if self.labels[i] == community_label:
                com_coh.append(self.cohesion_node(i))

        return np.mean(com_coh)

    def community_echo_chamber_index(self, community_label: int) -> float:
        com_eci = []

        for i in range(self.distances.shape[0]):
            if self.labels[i] == community_label:
                com_eci.append(self.metric(i))

        return np.mean(com_eci)

    def network_separation(self) -> float:
        net_sep = []

        for i in range(self.distances.shape[0]):
            net_sep.append(self.separation_node(i))

        return np.mean(net_sep)
