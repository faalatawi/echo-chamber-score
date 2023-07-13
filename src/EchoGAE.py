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
