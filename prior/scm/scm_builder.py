from typing import Dict, Optional, Union
import torch
import networkx as nx

from scm.scm import SCM
from mechanisms.mlp_mechanism import SampleMLPMechanism
from mechanisms.xg_boost_mechanism import SampleXGBoostMechanism
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
    xgboost_prob : float, default 0.1
        Probability of using XGBoost mechanism for each node (0.0 = never, 1.0 = always).
        Remaining nodes will use MLP mechanisms.
    node_dim : int
        The feature dimension of all nodes.
    
    # MLP Mechanism Hyperparameters
    mlp_num_hidden_layers : int, default 0
        Fixed number of hidden layers for MLP mechanisms.
    mlp_hidden_dim : int, default 16
        Width of hidden layers for MLP mechanisms.
    
    # XGBoost Mechanism Hyperparameters
    xgb_num_hidden_layers : int, default 0
        Number of hidden layers for XGBoost mechanisms (typically 0).
    xgb_hidden_dim : int, default 0
        Hidden dimension for XGBoost mechanisms (typically 0).
    xgb_n_training_samples : int, default 100
        Number of training samples for XGBoost mechanism fitting.
    xgb_add_noise : bool, default False
        Whether XGBoost mechanisms should add their own noise.
    
    # Noise Distribution Parameters
    root_std : float
        The standard deviation used to sample noise of root nodes.
    non_root_std : float
        The standard deviation used to sample noise of non-root nodes.
    
    # SCM Configuration
    scm_fast : bool, default True
        Whether to use fast sampling mode in the SCM.
    """
    
    def __init__(
        self,
        # the underlying graph
        graph: nx.DiGraph,
        *,
        # Mechanism Type Selection
        xgboost_prob: float = 0.1,
        node_dim: int = 1,
        
        # MLP Mechanism Hyperparameters
        mlp_num_hidden_layers: int = 0,
        mlp_hidden_dim: int = 16,
        
        # XGBoost Mechanism Hyperparameters
        xgb_num_hidden_layers: int = 0,
        xgb_hidden_dim: int = 0,
        xgb_n_training_samples: int = 100,
        xgb_add_noise: bool = False,
        
        # noise parameters
        root_std: float = 1.0,
        non_root_std: float = 0.1,
        
        # SCM Configuration
        scm_fast: bool = True,
    ) -> None:
        # Store all parameters
        self.graph = graph
        
        self.xgboost_prob = xgboost_prob
        self.node_dim = node_dim
        
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        
        self.xgb_num_hidden_layers = xgb_num_hidden_layers
        self.xgb_hidden_dim = xgb_hidden_dim
        self.xgb_n_training_samples = xgb_n_training_samples
        self.xgb_add_noise = xgb_add_noise
        
        self.root_std = root_std
        self.non_root_std = non_root_std
        
        self.scm_fast = scm_fast
        
        # Validate all hyperparameters
        self._validate_hyperparameters()
    
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
        scm = SCM(self.graph, mechanisms, noise, fast=self.scm_fast)
        
        return scm
    
    def _create_mechanisms(self, generator: Optional[torch.Generator]) -> Dict[int, Union[SampleMLPMechanism, SampleXGBoostMechanism]]:
        """Create mechanisms for each node in the DAG."""
        mechanisms = {}
        
        for node in self.graph.nodes():
            # Sample whether to use XGBoost or MLP
            # never use XGBoost if input dimension is 0
            input_dim = len(list(self.graph.predecessors(node))) * self.node_dim
            use_xgboost = torch.rand(1, generator=generator).item() < self.xgboost_prob and input_dim > 0
            
            if use_xgboost:
                mechanisms[node] = SampleXGBoostMechanism(
                    input_dim=input_dim,
                    node_dim=self.node_dim,
                    num_hidden_layers=self.xgb_num_hidden_layers,
                    hidden_dim=self.xgb_hidden_dim,
                    n_training_samples=self.xgb_n_training_samples,
                    generator=generator,
                    add_noise=self.xgb_add_noise
                )
            else:
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
    
    def _validate_hyperparameters(self) -> None:
        """
        Validate all hyperparameters for correctness and consistency.
        
        Raises
        ------
        ValueError
            If any hyperparameter is invalid or inconsistent.
        """
        
        # Mechanism type validation
        if not (0.0 <= self.xgboost_prob <= 1.0):
            raise ValueError(f"xgboost_prob must be in [0.0, 1.0], got {self.xgboost_prob}")
        
        if not isinstance(self.node_dim, int) or not self.node_dim > 0:
            raise ValueError(f"mlp_node_dim must be positive integer, got {self.node_dim}")
        
        # MLP mechanism validation
        valid_mlp_nonlins = {
            "mixed", "tabicl", "sophisticated_sampling_1", "sophisticated_sampling_1_normalization",
            "sophisticated_sampling_1_rescaling_normalization", "tanh", "sin", "neg", "id", "elu",
            "summed", "post", "linear"
        }
        
        if not isinstance(self.mlp_num_hidden_layers, int) or self.mlp_num_hidden_layers < 0:
            raise ValueError(f"mlp_num_hidden_layers must be non-negative integer, got {self.mlp_num_hidden_layers}")
        
        if not isinstance(self.mlp_hidden_dim, int) or self.mlp_hidden_dim < 1:
            raise ValueError(f"mlp_hidden_dim must be positive integer, got {self.mlp_hidden_dim}")        
        
        # XGBoost mechanism validation
        if not isinstance(self.xgb_num_hidden_layers, int) or self.xgb_num_hidden_layers < 0:
            raise ValueError(f"xgb_num_hidden_layers must be non-negative integer, got {self.xgb_num_hidden_layers}")
        
        if not isinstance(self.xgb_hidden_dim, int) or self.xgb_hidden_dim < 0:
            raise ValueError(f"xgb_hidden_dim must be non-negative integer, got {self.xgb_hidden_dim}")
        
        
        if not isinstance(self.xgb_n_training_samples, int) or self.xgb_n_training_samples < 1:
            raise ValueError(f"xgb_n_training_samples must be positive integer, got {self.xgb_n_training_samples}")
        
        if not isinstance(self.xgb_add_noise, bool):
            raise ValueError(f"xgb_add_noise must be boolean, got {self.xgb_add_noise}")
        
        # Noise distribution validation
        if self.root_std is None or not isinstance(self.root_std, (int, float)) or self.root_std <= 0:
            raise ValueError(f"root_std must be positive number, got {self.root_std}")
        
        if self.non_root_std is None or not isinstance(self.non_root_std, (int, float)) or self.non_root_std <= 0:
            raise ValueError(f"non_root_std must be positive number, got {self.non_root_std}")
        
        # SCM configuration validation
        if not isinstance(self.scm_fast, bool):
            raise ValueError(f"scm_fast must be boolean, got {self.scm_fast}")