from typing import Any, Dict, Mapping, Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
import networkx as nx

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
                parents_feat = torch.cat(parts, dim=2).to(device=self.device, dtype=self.dtype)
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
    
    torch.no_grad()
    def log_likelihood(self, sampled_values: Dict[Any, Tensor], y_var: str) -> float:
        """Compute the log-likelihood of the provided value of `y_var` conditioned on all other variables."""
        # Propagate sampled values through the SCM and compute noise
        sampled_noise = {}
        for v in self._topo:
            if self._is_root[v]:
                sampled_noise[v] = sampled_values[v]
            else:
                mech = self.mechanisms[v]
                parts = [sampled_values[p] for p in self._parents[v]]
                parents_feat = torch.cat(parts, dim=2).to(device=self.device, dtype=self.dtype)
                y = mech(parents_feat, eps=None)
                sampled_noise[v] = sampled_values[v] - y
        # compute log likelhood sample-wise
        shape = sampled_values[list(sampled_values.keys())[0]].shape
        total_log_likelihood = 0.0
        for idx in np.ndindex(shape):
            log_prob = 0.0
            for v in self._topo:
                log_prob += self.noise[v].log_prob(sampled_noise[v][idx])
            total_log_likelihood += log_prob
        return total_log_likelihood