import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community.louvain import louvain_communities
from sentence_transformers import SentenceTransformer
from echoey.content_embedding.tweet_preprocessing import preprocess_tweet_for_bert
from sklearn.utils import check_random_state
import torch
import random

import os

from tqdm import tqdm

tqdm.pandas()

# disable warnings
import warnings

warnings.filterwarnings("ignore")


def get_data(
    dataset_path: str, louvain_resolution=0.05, SBert_model_name="all-MiniLM-L6-v2"
):
    # Set the random seed for python
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    pd.np.random.seed(seed)
    check_random_state(seed)
    torch.manual_seed(seed)

    # 1: Get the Graph G
    G = nx.read_gml(dataset_path + "graph.gml")
    if nx.is_directed(G):
        G = G.to_undirected()

    # 2: Get the user embeddings
    df = pd.read_feather(dataset_path + "tweets.feather")

    if os.path.exists(dataset_path + "embeddings.feather"):
        # print("Loading embeddings")
        df_emb = pd.read_feather(dataset_path + "embeddings.feather")
        df = df.merge(df_emb, on="user_id", how="inner")
    else:
        # print("No embeddings found, creating them")
        def preprocess_tweets(tweets):
            out = []
            for tw in tweets:
                tw = preprocess_tweet_for_bert(tw)
                if len(tw) > 1:
                    out.append(" ".join(tw))
            return out

        # print("Preprocessing tweets")
        df["tweets"] = df.tweets.progress_apply(preprocess_tweets)
        # Remove users with no tweets
        df = df[df.tweets.apply(len) > 0]

        # model_name = "all-mpnet-base-v2"
        # model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(SBert_model_name)

        def embed_user_tweets(tweets):
            emb = model.encode(tweets)
            emb = np.mean(emb, axis=0)
            return emb

        # print("Embedding tweets")
        df["embeddings"] = df.tweets.progress_apply(embed_user_tweets)

        df_emb = df[["user_id", "embeddings"]]
        df_emb.reset_index(drop=True, inplace=True)
        df_emb.to_feather(dataset_path + "embeddings.feather")

    # 3: Filter the graph
    G = G.subgraph(df.user_id)
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes)
    df = df[df.user_id.isin(G.nodes())]

    # 5: Find the communities
    community = louvain_communities(G, resolution=louvain_resolution, seed=42)
    community = list(community)

    def which_community(node):
        for i, c in enumerate(community):
            if node in c:
                return i
        return -1

    df["community"] = df.user_id.apply(which_community)

    # allsides
    df_allsides = pd.read_feather(dataset_path + "allsides.feather")

    # 4: Make a map from user_id to index
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_id_map)

    # 6: Get the embeddings and labels
    users_embeddings = df.set_index("user_id")["embeddings"].to_dict()
    labels = df.set_index("user_id")["community"].to_dict()
    allsides_scores = df_allsides.set_index("user_id")["allsides_score"].to_dict()
    allsides_scores = {
        node_id_map[user_id]: score
        for user_id, score in allsides_scores.items()
        if user_id in node_id_map
    }

    users_embeddings_tmp = {}
    labels_tmp = {}

    for user_id, index in node_id_map.items():
        users_embeddings_tmp[index] = users_embeddings[user_id]
        labels_tmp[index] = labels[user_id]

    users_embeddings = users_embeddings_tmp
    labels = labels_tmp
    labels = np.array(list(labels.values()))

    return G, users_embeddings, labels, allsides_scores, node_id_map


def get_data_garimella(
    dataset: str, louvain_resolution=0.05, SBert_model_name="all-MiniLM-L6-v2"
):
    graph_path = f"garimella_datasets/clean_graphs/{dataset}.gml"
    user_info_path = f"garimella_datasets/users_info/{dataset}_users_info.feather"

    # 1: Get the Graph G
    G = nx.read_gml(graph_path)
    if nx.is_directed(G):
        G = G.to_undirected()

    # 2: Get the user embeddings
    df = pd.read_feather(user_info_path)

    df = df[df.description.str.len() > 0]
    df["username"] = df.username.str.lower()

    # 3: Filter the graph
    G = G.subgraph(df.username)
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes)
    df = df[df.username.isin(G.nodes())]

    # 4: Embedding
    df["description"] = df.description.apply(
        lambda x: " ".join(preprocess_tweet_for_bert(x))
    )

    model = SentenceTransformer(SBert_model_name)

    embeddings = model.encode(df.description.to_list(), show_progress_bar=False)
    df["embeddings"] = embeddings.tolist()

    # 5: Find the communities
    community = louvain_communities(G, resolution=louvain_resolution)
    community = list(community)

    def which_community(node):
        for i, c in enumerate(community):
            if node in c:
                return i
        return -1

    df["community"] = df.username.apply(which_community)

    # 4: Make a map from user_id to index
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_id_map)

    # 6: Get the embeddings and labels
    users_embeddings = df.set_index("username")["embeddings"].to_dict()
    labels = df.set_index("username")["community"].to_dict()

    users_embeddings_tmp = {}
    labels_tmp = {}

    for user_id, index in node_id_map.items():
        users_embeddings_tmp[index] = users_embeddings[user_id]
        labels_tmp[index] = labels[user_id]

    users_embeddings = users_embeddings_tmp
    labels = labels_tmp
    labels = np.array(list(labels.values()))

    return G, users_embeddings, labels
