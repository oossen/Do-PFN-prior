from typing import Dict, Optional, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx

from dopfnprior.utils.sampling import DistributionSampler
from dopfnprior.utils.likelihood_wrapper import LikelihoodWrapper


class SCM:
    """
    Structural Causal Model with vectorized ancestral sampling.

    Parameters
    ----------
    dag : nx.DiGraph[str]
        The DAG underlying the SCM.
        Each node should have a node feature `dimension`.
    mechanisms : Dict[str, nn.Module]
        The mechanisms for each node.
        Should be a *deterministic* function of parent values.
    noise : Dict[str, DistributionSampler]
        The noise distributions for each node.
        Must implement sampling of noise and log-probabilities for any input shape.
    post_activations : Dict[str, nn.Module]
        Optional post-activation functions for each node.
        If not provided, no post-activation is applied.
    
    Workflow
    --------
    1) scm.sample_noise(S)                 # samples and fixes noise (S is any shape)
    2) xs = scm.propagate()                # uses the fixed noises
    
    Denoting by `f_i`, `eps_i`, and `g_i` the mechanism, noise, and post-activation for node `i`, the SCM implements
    the structural causal model determined by the equations:
    
        x_i = g_i(f_i(parents(x_i)) + eps_i)
    
    When called with shape S = (s_1, ..., s_n), both the sampled noise and the propagated values have shape
    
        (s_1, ..., s_n, d_i),
        
    where d_i is the dimension of node i.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        mechanisms: Dict[str, nn.Module],
        noise: Dict[str, DistributionSampler],
        post_activations: Dict[str, nn.Module],
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.post_activations = post_activations

        # Topology and parents
        self._topo: List[str] = list(nx.topological_sort(dag))
        self._parents: Dict[str, List[str]] = {v: list(self.dag.predecessors(v)) for v in self._topo}

        # Fixed noise buffers
        self._sampled_noise: Dict[str, Tensor] = {}
        
    def __str__(self) -> str:
        info = []
        info.append(f"SCM with nodes: {list(self.dag.nodes())}")
        for v in self._topo:
            info.append(v)
            info.append(f"Parents: {self._parents[v]}")
            info.append(f"Mechanism: {self.mechanisms[v]}")
            info.append(f"Noise std: {self.noise[v].std()}")
        return "\n".join(info)
    
    @torch.no_grad()
    def sample_noise(self, sample_shape: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> None:
        """
        Sample and save noise for each node in the SCM.
        
        Parameters
        ----------
        sample_shape : tuple of int
            The shape of the data to be sampled for each feature.
            If a node has dimension d, the sampled noise for that node will have shape `sample_shape + (d,)`.
        generator : torch.Generator, optional
            To make the sampling process reproducible.
        """
        
        for v in self._topo:
            dist_v = self.noise[v]
            sample_shape_v = sample_shape + (self.dag.nodes[v].get("dimension", 1),)
            eps_v = dist_v.sample_shape(sample_shape_v, generator=generator)
            self._sampled_noise[v] = eps_v

    @torch.no_grad()
    def propagate(self) -> Dict[str, Tensor]:
        """
        Propagate through the SCM using the fixed noise sampled by `sample_noise`.
        Returns a dictionary mapping each node to its sampled value.
        All values have the same shape as their corresponding noise.
        """
        xs: Dict[str, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {}
            for p in self._parents[v]:
                parents_feat[p] = xs[p]
            eps_v = self._sampled_noise[v]
            x = mech(parents_feat) + eps_v
            if v in self.post_activations:
                x = self.post_activations[v](x)
            xs[v] = x

        return 
    
    @torch.no_grad()
    def total_log_probability(self, values: Dict[str, Tensor]) -> Tensor:
        """
        Compute the probabilities of the provided values under the SCM.
        The returned tensor has the same shape as the input values.
        """
        sampled_noise = {}
        value_shape = list(values.values())[0].shape
        log_prob = torch.zeros(value_shape)
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {p: values[p] for p in self._parents[v]}
            x = mech(parents_feat)
            sampled_noise[v] = values[v] - x
            log_prob += self.noise[v].log_prob(sampled_noise[v])
        return log_prob
    
    @torch.no_grad()
    def log_likelihood_batch(self, values: Dict[str, Tensor], y_values: Tensor, y_var: str = 'y') -> Tensor:
        
        def mechanism_fn(values: Dict[str, Tensor]) -> Dict[str, Tensor]:
            noise_free_values = {}
            for v in self._topo:
                mech = self.mechanisms[v]
                parents_feat = {p: values[p] for p in self._parents[v]}
                x = mech(parents_feat)
                noise_free_values[v] = x
            return noise_free_values
        
        def noise_fn(noise_residuals: Dict[str, Tensor]) -> Tensor:
            log_probs = [self.noise[v].log_prob(noise_residuals[v]).sum(dim=-1) for v in self._topo]
            return sum(log_probs)
        
        wrapper = LikelihoodWrapper(mechanism_fn, noise_fn)
        return wrapper.log_likelihood_batch(values, y_values, y_var)