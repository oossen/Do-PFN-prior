from typing import Optional

import numpy as np
import networkx as nx
import torch


class GraphBuilder:
    """
    Utility class for generating random DAGs (Directed Acyclic Graphs).
    Acyclicity is ensured by sampling edges only from earlier to later nodes in
    a random topological order (random permutation).
    """

    def __init__(self, num_nodes: int, 
                 edge_prob: float,  
                 dropout_prob: float,
                 features_per_node_dist) -> None:
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        edge_prob : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        features_per_node : int
            The dimension of each node in the graph.
        dropout_prob : float
            Probability of making a given node hidden.
        """
        self.num_nodes = num_nodes
        # Set a minimum probability to avoid very sparse small graphs
        # 2 -> 87%, 3 -> 54%, 5 -> 28%, 10 -> 13%, 20 -> 5% 30 -> 3%
        edge_prob_min = 2 / (num_nodes ** 1.2)
        self.edge_prob = max(edge_prob_min, edge_prob) 
        self.dropout_prob = dropout_prob
        self.features_per_node_dist = features_per_node_dist
    
    def sample(self, generator: Optional[torch.Generator]) -> nx.DiGraph:
        """
        Create a random DAG.

        Parameters
        ----------
        generator : torch.Generator
            Used to make sampling of graphs deterministic.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        """
        # Get numpy generator from torch generator
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        self.rng = np.random.default_rng(np_seed)
        
        n = int(self.num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1].")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Random topological order
        perm = self.rng.permutation(n)

        # Strictly upper-triangular Bernoulli mask (acyclic by construction)
        mask = np.triu(self.rng.random((n, n)) < self.edge_prob, k=1)

        # Extract and add edges
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = perm[i_idx]
            dst = perm[j_idx]
            G.add_edges_from(zip(src.tolist(), dst.tolist()))
            
        # resample if there are no edges
        if len(G.edges) == 0:
            return self.sample(generator)
        
        # Hide some features and set node dimensions
        hidden_dict = {}
        visible_dict = {}
        dim_dict = {}
        for v in G.nodes():
            node_dim = self.features_per_node_dist.sample(generator)
            dim_dict[v] = node_dim
            n_hidden = self.rng.binomial(node_dim, self.dropout_prob)
            n_visible = node_dim - n_hidden
            hidden_dict[v] = n_hidden
            visible_dict[v] = n_visible
        nx.set_node_attributes(G, dim_dict, name="dimension")
        nx.set_node_attributes(G, hidden_dict, name="n_hidden")
        nx.set_node_attributes(G, visible_dict, name="n_visible")
        
        # select target
        target_node = list(G.nodes())[-1]
        # resample if target has no parents or no children
        y_has_pred = len([v for v in list(G.predecessors(target_node)) if G.nodes()[v]["n_visible"] > 0]) > 0
        y_has_succ = len([v for v in list(G.successors(target_node)) if G.nodes()[v]["n_visible"] > 0]) > 0
        if not y_has_pred or not y_has_succ:
            return self.sample(generator)
        # resample if target node has no visible features
        if G.nodes()[target_node]["n_visible"] == 0:
            return self.sample(generator)

        # rename nodes
        renaming = {}
        for v in G.nodes():
            if v != target_node:    
                renaming[v] = f"x{str(v)}"
        renaming[target_node] = "y"
        G = nx.relabel_nodes(G, renaming)
        
        return G

    