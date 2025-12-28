from typing import Any, Dict, Optional
import torch
import torch.distributions as dist
import networkx as nx

from dopfnprior.scm.scm import SCM
from dopfnprior.mechanisms.simple_mechanism import SimpleMechanism
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
        # noise parameters
        root_std: float = 1.0,
        non_root_std: float = 0.1,
        root_mean: float = 0.0,
        non_root_mean: float = 0.0,
    ) -> None:
        # Store all parameters
        self.graph = graph
        self.mean_root_std = root_std
        self.mean_non_root_std = non_root_std
        self.root_mean = root_mean if root_mean is not None else 0.0
        self.non_root_mean = non_root_mean if non_root_mean is not None else 0.0
        self.root_std = {}
        self.non_root_std = {}
    
    def sample(self, generator: torch.Generator) -> SCM:
        """
        Build and return a configured SCM based on the provided hyperparameters.
        
        Returns
        -------
        SCM
            A fully configured Structural Causal Model ready for sampling.
        """
        # Step 1: Create mechanisms for each node
        if not hasattr(self, 'mechanisms'):
            nodes = self.graph.nodes
            self.mechanisms = {v: SimpleMechanism(list(nodes), generator) for v in nodes}
        
        # Step 2: Create noise distributions
        # Note that creation of the distributions is deterministic and requires no generator
        if not hasattr(self, 'noise'):
            self.noise = self._create_noise_distribution(generator)
        
        # Step 3: Build the SCM
        scm = SCM(self.graph, self.mechanisms, self.noise, generator)
        
        return scm
    
    def _create_noise_distribution(self, generator: Optional[torch.Generator]) -> Dict[Any, TorchDistributionSampler]:
        """Create noise distributions for exogenous and endogenous variables."""
        root_nodes = [v for v in self.graph.nodes() if not self.graph.predecessors(v)]
        non_root_nodes = [v for v in self.graph.nodes() if self.graph.predecessors(v)]
        root_std_gen = TorchDistributionSampler(dist.Exponential(rate=1/self.mean_root_std))
        non_root_std_gen = TorchDistributionSampler(dist.Exponential(rate=1/self.mean_non_root_std))
        
        noise = {}
        for v in root_nodes:
            self.root_std[v] = root_std_gen.sample(generator)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.root_mean, scale=self.root_std[v]))
        for v in non_root_nodes:
            self.non_root_std[v] = non_root_std_gen.sample(generator)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.non_root_mean, scale=self.non_root_std[v]))
        return noise