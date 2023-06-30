import numpy as np
import networkx as nx
import pandas as pd


# PyThorch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

# Typing
from typing import Tuple
from pandas import DataFrame
from networkx import Graph


from .GAE import run


def read_data(
    graph_path: str,
    tweet_embedding_path: str,
    labels_path: str,
) -> Tuple[Graph, DataFrame, DataFrame]:
    """
    Read the data from the paths
    Input:
        graph_path: Path to the graph (as a gml file)
        tweet_embedding_path: Path to the tweet embeddings (as a pickle file)
        labels_path: Path to the labels (as a feather file)

    Output:
        G: The graph
        df_embeddings: The tweet embeddings
        df_labels: The labels
    """

    df_embeddings = pd.read_pickle(tweet_embedding_path)

    G = nx.read_gml(graph_path)

    df_labels = pd.read_feather(labels_path)

    return G, df_embeddings, df_labels


def preprocess(
    G: Graph, df_embeddings: DataFrame, df_labels: DataFrame
) -> Tuple[Graph, dict, dict]:
    # 1: Users Embeddings
    choice = np.random.choice
    MAX_TWEETS = 50

    average_embeddings = lambda x: x[
        choice(x.shape[0], min(MAX_TWEETS, x.shape[0]), replace=False)
    ].mean(axis=0)

    df_embeddings["user_emb"] = df_embeddings.embeddings.apply(average_embeddings)

    users_embeddings = df_embeddings.set_index("user")["user_emb"].to_dict()

    # ---------------
    # 2 Graph
    # Make the Graph undirected
    if G.is_directed():
        G = nx.to_undirected(G)

    # Limit the graph to the users with embeddings
    users_with_embeddings = list(users_embeddings.keys())
    G = G.subgraph(users_with_embeddings)

    # Get largest connected component
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes)

    # ---------------
    # 3 Labels
    df_labels = df_labels[["user_id", "allsides_score"]]
    # Set the index to user_id
    df_labels = df_labels.set_index("user_id")
    # To dict
    ground_truth = df_labels.to_dict()["allsides_score"]

    # ---------------
    # 4: Relabel the nodes to be integers
    # This is needed because VGAE expects the nodes to be labeled with integers

    node_id_map = {node: i for i, node in enumerate(G.nodes())}

    # Replace user_id with index(int) using the map
    G = nx.relabel_nodes(G, node_id_map)

    ground_truth = {
        node_id_map[u]: ground_truth[u] for u in ground_truth if u in node_id_map
    }

    users_embeddings_tmp = {}
    for user_id, index in node_id_map.items():
        users_embeddings_tmp[index] = users_embeddings[user_id]
    users_embeddings = users_embeddings_tmp

    return G, users_embeddings, ground_truth


def EchoGAE_algorithm(
    G,
    user_embeddings=None,
    show_progress=True,
    epochs=300,
    hidden_channels=100,
    out_channels=50,
) -> np.ndarray:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create node features
    if user_embeddings is None:
        X = torch.eye(len(G.nodes), dtype=torch.float32, device=DEVICE)
    else:
        X = []
        for node in G.nodes:
            X.append(user_embeddings[node])
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # Create edge list
    edge_list = np.array(G.edges).T
    edge_list = torch.tensor(edge_list, dtype=torch.int64).to(DEVICE)

    # Data object
    data = Data(x=X, edge_index=edge_list)
    data = train_test_split_edges(data)

    # Run the model
    model, x, train_pos_edge_index = run(
        data,
        show_progress=show_progress,
        epochs=epochs,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    # Embedding
    GAE_embedding = model.encode(x, train_pos_edge_index).detach().cpu().numpy()

    return GAE_embedding
