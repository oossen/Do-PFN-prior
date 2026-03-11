from typing import Dict, Optional, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx

from tfmplayground.utils import get_default_device

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
    device : torch.device
        The device on which the model's mechanisms live.
    
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
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else get_default_device()
        self.dag = dag
        self.mechanisms = {v: mechanisms[v].to(self.device) for v in mechanisms}
        self.noise = noise
        self.post_activations = {v: post_activations[v].to(self.device) for v in post_activations}

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
            if v in self.mechanisms:
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
            self._sampled_noise[v] = eps_v.to(self.device)

    @torch.no_grad()
    def propagate(self) -> Dict[str, Tensor]:
        """
        Propagate through the SCM using the fixed noise sampled by `sample_noise`.
        Returns a dictionary mapping each node to its sampled value.
        All values have the same shape as their corresponding noise.
        """
        xs: Dict[str, Tensor] = {}
        for v in self._topo:
            x = self._sampled_noise[v]
            parents_feat = {p: xs[p] for p in self._parents[v]}
            if len(parents_feat) > 0:
                mech = self.mechanisms[v]
                x = mech(parents_feat) + x
            if v in self.post_activations:
                x = self.post_activations[v](x)
            xs[v] = x

        return xs
    
    
    @torch.no_grad()
    def log_likelihood_batch(self, 
                             values: Dict[str, Tensor], 
                             y_values: Tensor, 
                             y_var: str = 'y',
                             y_idx: int = 0,
                             plot_dir: Optional[str] = None) -> Tensor:
        
        noise_dists = {v: lambda eps : self.noise[v].log_prob(eps).sum(dim=-1) for v in self._topo}
        wrapper = LikelihoodWrapper(self.dag, self.mechanisms, noise_dists)
        return wrapper.log_likelihood_batch(values, y_values, y_var, y_idx, plot_dir)