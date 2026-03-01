import networkx as nx
import random
from datetime import datetime, timedelta

def simulate_propagation():
    G = nx.DiGraph()

    # Simulate users
    users = [f"User_{i}" for i in range(1, 15)]

    origin = random.choice(users)
    G.add_node(origin)

    # Simulate spreading
    for _ in range(20):
        u = random.choice(users)
        v = random.choice(users)
        if u != v:
            G.add_edge(u, v)

    return G, origin


def analyze_propagation(G):
    degree_centrality = nx.degree_centrality(G)
    max_centrality = max(degree_centrality.values())

    if max_centrality > 0.5:
        return 0.9  # suspicious spread
    else:
        return 0.4  # normal spread