import math
from typing import Any, Dict, Mapping, Optional, Tuple, List, cast
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx
from scipy.integrate import quad
from scipy.optimize import minimize_scalar



class SCM:
    """
    Structural Causal Model with vectorized ancestral sampling.

    Workflow
    --------
    1) scm.sample(B)                        # samples & fixes noise
    2) xs = scm.propagate(B)                # uses the fixed noises

    Parameters
    ----------
    dag : CausalDAG
    mechanisms : Mapping[str, BaseMechanism]
    noise : Mapping[int, Distribution]
    device : torch.device | str
    dtype : torch.dtype
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        mechanisms: Mapping[Any, nn.Module],
        noise: Mapping,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = torch.device(device)
        self.dtype = dtype

        # Topology & parents
        self._topo: List = list(nx.topological_sort(dag))
        self._parents: Dict[Any, List] = {v: list(self.dag.predecessors(v)) for v in self._topo}
        self._is_root: Dict[Any, bool] = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # Fixed noise buffers
        self._sampled_noise: Dict[Any, Tensor] = {}
        
        # Fit normalization
        n_noise_fitting_samples = 100
        self.sample_noise((n_noise_fitting_samples,), generator=generator)
        self.propagate((n_noise_fitting_samples,))
    
    @torch.no_grad()
    def sample_noise(self,
                     sample_shape: Tuple[int, ...],
                     *,
                     generator: Optional[torch.Generator] = None,
                     nodes: Optional[List] = None
                     ) -> Dict[Any, Tensor]:
        """
        Sample & fix noise (eps) for all nodes.
        If `nodes` is provided, resample only those nodes.
        """
        target_nodes = nodes if nodes is not None else self._topo
        views: Dict[Any, Tensor] = {}
        for v in target_nodes:
            dist_v = self.noise.get(v, None)
            e_v = dist_v.sample_shape(sample_shape, generator=generator)
            if not isinstance(e_v, Tensor):
                e_v = torch.as_tensor(e_v)
            views[v] = e_v

        self._sampled_noise = views
        return views

    @torch.no_grad()
    def propagate(self, sample_shape: Tuple[int, ...]) -> Dict[Any, Tensor]:
        xs: Dict[Any, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {v: xs[p] for p in self._parents[v]}
            eps_v = self._sampled_noise[v].to(device=self.device, dtype=self.dtype) if v in self._sampled_noise else None

            x = mech(parents_feat, eps=eps_v)
            xs[v] = x

        # only return data for non-hidden nodes
        xs = {v: x for v, x in xs.items() if not self.dag.nodes[v].get("hidden", False)}
        return xs
    
    @torch.no_grad()
    def log_likelihood(self, values: Dict[Any, Tensor], y_var: str) -> float:
        """Compute the log-likelihood of the provided value of `y_var` conditioned on all other variables."""          
        shape = values[list(values.keys())[0]].shape
        total_log_likelihood = 0.0
        
        for idx in np.ndindex(shape):
            values_i = {v: values[v][idx] for v in values}
            joint_log_likelihood = self.total_log_probability(values_i)
            
            def integrand(y):
                values_i[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
                log_prob = self.total_log_probability(values_i)
                return math.exp(log_prob)
            
            # find maximum and integrate around it
            res = minimize_scalar(lambda y: -integrand(y))
            maximum: float = cast(float, res.x)
            a, b = maximum - 10.0, maximum + 10.0
            marginal = quad(integrand, a, b, points=[maximum])[0]
            eps = 1e-12  # to avoid log(0)
            log_marginal = math.log(marginal + eps)
            total_log_likelihood += joint_log_likelihood - log_marginal

        return total_log_likelihood
    
    @torch.no_grad()
    def total_log_probability(self, values: Dict[Any, Tensor]) -> float:
        """
        Compute the probability of the provided values under the SCM,
        using the mechanisms saved in `self.mechanisms`.
        """
        log_prob = 0.0
        sampled_noise = {}
        for v in self._topo:
            if self._is_root[v]:
                sampled_noise[v] = values[v]
            else:
                mech = self.mechanisms[v]
                parents_feat = {v: values[p] for p in self._parents[v]}
                x = mech(parents_feat, eps=None)
                sampled_noise[v] = values[v] - x
            log_prob += self.noise[v].log_prob(sampled_noise[v])
        return log_prob