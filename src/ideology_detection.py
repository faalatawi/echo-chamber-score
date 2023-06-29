import numpy as np
import networkx as nx
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def ideology_detection(
    user_embeddings: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    distances = None
    cluster_label = None

    if metric == "cosine":
        distances = cosine_distances(user_embeddings)

        # TODO: Lean about linkage methods
        cluster_label = AgglomerativeClustering(
            n_clusters=2, affinity="cosine", linkage="average"
        ).fit_predict(user_embeddings)

    elif metric == "euclidean":
        distances = euclidean_distances(user_embeddings)

        cluster_label = AgglomerativeClustering(
            n_clusters=2, affinity="euclidean", linkage="ward"
        ).fit_predict(user_embeddings)
    else:
        raise ValueError(f"Unknown metric {metric}")

    user_ideology_scores = []
    for i in range(len(distances)):
        user_ideology_scores.append(
            distances[i][cluster_label == 0].mean()
            - distances[i][cluster_label == 1].mean()
        )
    user_ideology_scores = np.array(user_ideology_scores)

    # Normalize the polarization between -1 and 1
    min_score = user_ideology_scores.min()
    max_score = user_ideology_scores.max()

    user_ideology_scores = (user_ideology_scores - min_score) / (max_score - min_score)
    user_ideology_scores = user_ideology_scores * 2 - 1

    return user_ideology_scores
