from typing import Any, Dict, Literal, Optional, Tuple, overload
from pyparsing import Union
import torch
import torch.distributions as dist
import networkx as nx

from dopfnprior.scm.scm import SCM
from dopfnprior.mechanisms.mlp_mechanism import MLPMechanism
from dopfnprior.utils.sampling import TorchDistributionSampler


class SCMBuilder:
    """
    Builder class for creating Structural Causal Models (SCMs) with configurable hyperparameters.
    
    This class provides a comprehensive interface for building SCMs with various mechanism types,
    noise distributions, and graph structures.
    
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
        The mean standard deviation used to sample noise of root nodes.
    non_root_std : float
        The mean standard deviation used to sample noise of non-root nodes.
    root_mean : float
        The mean use to sample noise of root nodes.
    non_root_mean : float
        The mean used to sample noise of non-root nodes.
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
        root_mean: float = 0.0,
        non_root_mean: float = 0.0,
    ) -> None:
        # Store all parameters
        self.graph = graph
        self.node_dim = node_dim
        
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        
        self.root_std = root_std
        self.non_root_std = non_root_std
        self.root_mean = root_mean if root_mean is not None else 0.0
        self.non_root_mean = non_root_mean if non_root_mean is not None else 0.0
    
    @overload
    def sample(self, generator: torch.Generator, return_log_prob: Literal[False] = False) -> SCM: ...
    
    @overload
    def sample(self, generator: torch.Generator, return_log_prob: Literal[True] = True) -> Tuple[SCM, float]: ...
    
    def sample(self, generator: torch.Generator, return_log_prob: bool = False) -> Union[SCM, Tuple[SCM, float]]:
        """
        Build and return a configured SCM based on the provided hyperparameters.
        
        Returns
        -------
        SCM
            A fully configured Structural Causal Model ready for sampling.
        """
        # Step 1: Create mechanisms for each node
        if not hasattr(self, 'mechanisms'):
            self.mechanisms, self.log_prob_mechanisms = self._create_mechanisms(generator)
        
        # Step 2: Create noise distributions
        # Note that creation of the distributions is deterministic and requires no generator
        if not hasattr(self, 'noise'):
            self.noise, self.log_prob_noise = self._create_noise_distribution(generator)
        
        # Step 3: Build the SCM
        scm = SCM(self.graph, self.mechanisms, self.noise)
        
        if return_log_prob:
            total_log_prob = self.log_prob_mechanisms + self.log_prob_noise
            return scm, total_log_prob
        else:
            return scm
    
    def _create_mechanisms(self, generator: Optional[torch.Generator]) -> Tuple[Dict[Any, MLPMechanism], float]:
        """Create mechanisms for each node in the DAG."""
        mechanisms = {}
        log_prob = 0.0
        for node in self.graph.nodes():
            input_dim = len(list(self.graph.predecessors(node))) * self.node_dim
            mechanisms[node] = MLPMechanism(
                input_dim=input_dim,
                node_dim=self.node_dim,
                num_hidden_layers=self.mlp_num_hidden_layers,
                hidden_dim=self.mlp_hidden_dim,
                generator=generator,
            )
            log_prob += mechanisms[node].log_prob()
        
        return mechanisms, log_prob
    
    def _create_noise_distribution(self, generator: Optional[torch.Generator]) -> Tuple[Dict[Any, TorchDistributionSampler], float]:
        """Create noise distributions for exogenous and endogenous variables."""
        root_nodes = [v for v in self.graph.nodes() if not self.graph.predecessors(v)]
        non_root_nodes = [v for v in self.graph.nodes() if self.graph.predecessors(v)]
        root_std_gen = TorchDistributionSampler(dist.Exponential(rate=1/self.root_std))
        non_root_std_gen = TorchDistributionSampler(dist.Exponential(rate=1/self.non_root_std))
        
        noise = {}
        log_prob = 0.0
        for v in root_nodes:
            std = root_std_gen.sample(generator)
            log_prob += root_std_gen.log_prob(std)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.root_mean, scale=std))
        for v in non_root_nodes:
            std = non_root_std_gen.sample(generator) 
            log_prob += non_root_std_gen.log_prob(std)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.non_root_mean, scale=std))

        return noise, log_prob