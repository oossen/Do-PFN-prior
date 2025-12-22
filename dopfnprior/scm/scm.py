import math
from typing import Any, Dict, Mapping, Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
import networkx as nx
from scipy.integrate import quad

from dopfnprior.mechanisms.base_mechanism import BaseMechanism


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
        mechanisms: Mapping[Any, BaseMechanism],
        noise: Mapping,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = torch.device(device)
        self.dtype = dtype

        # --- Topology & parents
        self._topo: List = list(nx.topological_sort(dag))
        self._parents: Dict[Any, List] = {v: list(self.dag.predecessors(v)) for v in self._topo}
        self._is_root: Dict[Any, bool] = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # --- Node dimensions
        self._node_dims: Dict[Any, int] = {}
        for v in self._topo:
            self._node_dims[v] = self.mechanisms[v].node_dim

        # --- Fixed noise buffers
        self._sampled_noise: Dict[Any, Tensor] = {}
    
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
            dv = self._node_dims[v]
            dist_v = self.noise.get(v, None)
            e_v = dist_v.sample_shape(sample_shape + (dv,), generator=generator)
            if not isinstance(e_v, Tensor):
                e_v = torch.as_tensor(e_v)
            views[v] = e_v

        self._sampled_noise = views
        return views
    
    @torch.no_grad()
    def set_root_values(self, assignments: Dict[Any, Tensor]):
        for v, value in assignments.items():
            assert v in self.dag.nodes, "Invalid node!"
            self._sampled_noise[v] = value

    @torch.no_grad()
    def propagate(self, sample_shape: Tuple[int, ...]) -> Dict[Any, Tensor]:
        xs: Dict[Any, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parts = [xs[p] for p in self._parents[v]]
            if len(parts) > 0:
                parents_feat = torch.cat(parts, dim=-1).to(device=self.device, dtype=self.dtype)
            else:
                parents_feat = torch.empty(sample_shape + (0,)) # this tensor has no elements

            eps_v = None
            if v in self._sampled_noise:
                eps_v = self._sampled_noise[v].to(device=self.device, dtype=self.dtype)

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
                log_prob = 0.0
                values_i[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
                log_prob = self.total_log_probability(values_i)
                return math.exp(log_prob)
            
            marginal = quad(integrand, -np.inf, np.inf)[0]
            log_marginal = math.log(marginal)
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
                parents_feat = torch.tensor([values[p] for p in self._parents[v]])
                x = mech(parents_feat, eps=None)
                sampled_noise[v] = values[v] - x
                log_prob += self.noise[v].log_prob(sampled_noise[v])
        return log_prob