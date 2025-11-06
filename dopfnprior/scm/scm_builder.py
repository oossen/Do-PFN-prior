from typing import Dict, Optional, Union
import torch
import networkx as nx

from scm.scm import SCM
from mechanisms.mlp_mechanism import SampleMLPMechanism
from scm.noise_dist import MixedDist


class SCMBuilder:
    """
    Builder class for creating Structural Causal Models (SCMs) with configurable hyperparameters.
    
    This class provides a comprehensive interface for building SCMs with various mechanism types,
    noise distributions, and graph structures. All hyperparameters are explicitly specified in
    the constructor rather than being sampled randomly.
    
    Constructor parameters
    ----------------------
    graph: nx.DiGraph
        The directed acyclic graph underlying this SCM.
    node_dim : int
        The feature dimension of all nodes.
    
    # MLP Mechanism Hyperparameters
    mlp_num_hidden_layers : int, default 0
        Fixed number of hidden layers for MLP mechanisms.
    mlp_hidden_dim : int, default 16
        Width of hidden layers for MLP mechanisms.
    
    # Noise Distribution Parameters
    root_std : float
        The standard deviation used to sample noise of root nodes.
    non_root_std : float
        The standard deviation used to sample noise of non-root nodes.
    """
    
    def __init__(
        self,
        # the underlying graph
        graph: nx.DiGraph,
        *,
        # the dimension of each node
        node_dim: int = 1,
        
        # MLP Mechanism Hyperparameters
        mlp_num_hidden_layers: int = 0,
        mlp_hidden_dim: int = 16,
        
        # noise parameters
        root_std: float = 1.0,
        non_root_std: float = 0.1,
    ) -> None:
        # Store all parameters
        self.graph = graph
        self.node_dim = node_dim
        
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        
        self.root_std = root_std
        self.non_root_std = non_root_std
    
    def build(self, generator: torch.Generator) -> SCM:
        """
        Build and return a configured SCM based on the provided hyperparameters.
        
        Returns
        -------
        SCM
            A fully configured Structural Causal Model ready for sampling.
        """
        # Step 1: Create mechanisms for each node
        mechanisms = self._create_mechanisms(generator)
        
        # Step 2: Create noise distributions
        # Note that creation of the distributions is deterministic and requires no generator
        noise = self._create_noise_distribution()
        
        # Step 3: Build the SCM
        scm = SCM(self.graph, mechanisms, noise)
        
        return scm
    
    def _create_mechanisms(self, generator: Optional[torch.Generator]) -> Dict[int, SampleMLPMechanism]:
        """Create mechanisms for each node in the DAG."""
        mechanisms = {}
        
        for node in self.graph.nodes():
            input_dim = len(list(self.graph.predecessors(node))) * self.node_dim
            mechanisms[node] = SampleMLPMechanism(
                input_dim=input_dim,
                node_dim=self.node_dim,
                num_hidden_layers=self.mlp_num_hidden_layers,
                hidden_dim=self.mlp_hidden_dim,
                generator=generator,
            )
        
        return mechanisms
    
    def _create_noise_distribution(self) -> Dict[int, MixedDist]:
        """Create noise distributions for exogenous and endogenous variables."""
        root_nodes = [v for v in self.graph.nodes() if not self.graph.predecessors(v)]
        non_root_nodes = [v for v in self.graph.nodes() if self.graph.predecessors(v)]
        noise = {v: MixedDist(std=self.root_std) for v in root_nodes} | {v: MixedDist(std=self.non_root_std) for v in non_root_nodes}
        return noise