from typing import Dict, Optional, List
import torch
import torch.distributions as dist
import torch.nn as nn
import networkx as nx

from dopfnprior.scm.scm import SCM
from dopfnprior.scm.simple_mechanism import SimpleMechanism
from dopfnprior.utils.sampling import TorchDistributionSampler, DistributionSampler, CategoricalSampler


class SCMBuilder:
    """
    Builder class for creating Structural Causal Models (SCMs) with configurable hyperparameters.
    
    Constructor parameters
    ----------------------
    graph: nx.DiGraph
        The directed acyclic graph underlying this SCM.
    activation_dist : CategoricalSampler
        Sampler for choosing a mechanism's activation.
    root_std_dist : DistributionSampler
        Distribution sampler for standard deviations of root node noise.
    non_root_std_dist : DistributionSampler
        Distribution sampler for standard deviations of non-root node noise.
    root_mean : float
        The mean use to sample noise of root nodes.
    non_root_mean : float
        The mean used to sample noise of non-root nodes.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        activation_dist: CategoricalSampler,
        root_std_dist: DistributionSampler,
        non_root_std_dist: DistributionSampler,
        root_mean: float = 0.0,
        non_root_mean: float = 0.0,
    ) -> None:
        # Store all parameters
        self.graph = graph
        self.activation_dist = activation_dist
        self.root_std_dist = root_std_dist
        self.non_root_std_dist = non_root_std_dist
        self.root_mean = root_mean
        self.non_root_mean = non_root_mean
    
    def sample(self, generator: Optional[torch.Generator]) -> SCM:
        """Build and return a configured SCM based on the provided hyperparameters."""
        # Step 1: Create mechanisms for each node
        self.mechanisms = self._create_mechanisms(generator)
        
        # Step 2: Create noise distributions
        # Note that creation of the distributions is deterministic and requires no generator
        self.noise = self._create_noise_distribution(generator)
        
        # Step 3: Build the SCM
        scm = SCM(self.graph, self.mechanisms, self.noise, post_activations={})
        
        return scm
    
    def _create_mechanisms(self, generator: Optional[torch.Generator]) -> Dict[str, nn.Module]:
        nodes = list(self.graph.nodes)
        mechanisms = {}
        for v in [v for v in nodes if self.graph.in_degree(v) > 0]:  # only create mechanisms for non-root nodes
            activation_class = self.activation_dist.sample(generator)
            node_dims = {v: self.graph.nodes[v].get("dimension", 1) for v in nodes}
            mech = SimpleMechanism(v, node_dims, activation_class, generator)
            mechanisms[v] = mech
        return mechanisms
    
    def _create_noise_distribution(self, generator: Optional[torch.Generator]) -> Dict[str, DistributionSampler]:
        """Create noise distributions for exogenous and endogenous variables."""
        root_nodes = [v for v in self.graph.nodes() if self.graph.in_degree(v) == 0]
        non_root_nodes = [v for v in self.graph.nodes() if self.graph.in_degree(v) > 0]
        
        noise = {}
        for v in root_nodes:
            std = self.root_std_dist.sample(generator)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.root_mean, scale=std))
        for v in non_root_nodes:
            std = self.non_root_std_dist.sample(generator)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.non_root_mean, scale=std))
        return noise