from typing import Any, Dict, Mapping, Optional, Tuple, List
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

    Fast vs Safe
    ------------
    - fast=True : no checks/casts, calls mech._forward directly.
    - fast=False: checks acyclicity and if mechanisms match the underlying graph.

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
        mechanisms: Mapping[int, BaseMechanism],
        noise: Mapping[int, Any],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = torch.device(device)
        self.dtype = dtype

        # --- Topology & parents
        self._topo: List[int] = list(nx.topological_sort(dag))
        self._parents: Dict[int, List[int]] = {v: list(self.dag.predecessors(v)) for v in self._topo}
        self._is_root: Dict[int, bool] = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # --- Node dimensions
        self._node_dims: Dict[int, int] = {}
        for v in self._topo:
            self._node_dims[v] = self.mechanisms[v].node_dim

        # --- Fixed noise buffers & per-node views
        self._fixed: Optional[Dict[int, Tensor]] = None
        self._fixed_shape: Optional[Tuple[int, ...]] = None
    
    @torch.no_grad()
    def sample_noise(self,
                     sample_shape: Tuple[int, ...],
                     *,
                     generator: Optional[torch.Generator] = None,
                     nodes: Optional[List[int]] = None
                     ) -> Dict[int, Tensor]:
        """
        Sample & fix noise (eps) for all nodes.
        If `nodes` is provided, resample only those nodes.
        """
        target_nodes = nodes if nodes is not None else self._topo
        views: Dict[int, Tensor] = {}
        for v in target_nodes:
            dv = self._node_dims[v]
            dist_v = self.noise.get(v, None)
            e_v = dist_v.sample_shape(sample_shape + (dv,), generator=generator)
            if not isinstance(e_v, Tensor):
                e_v = torch.as_tensor(e_v)
            views[v] = e_v

        self._fixed = views
        self._fixed_shape = sample_shape
        return views

    @torch.no_grad()
    def propagate(self, sample_shape: Tuple[int, ...]) -> Dict[int, Tensor]:
        xs: Dict[int, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parts = [xs[p] for p in self._parents[v]]
            if len(parts) > 0:
                parents_feat = torch.cat(parts, dim=2).to(device=self.device, dtype=self.dtype)
            else:
                parents_feat = torch.empty(sample_shape + (0,)) # this tensor has no elements

            eps_v = None
            if self._fixed is not None and v in self._fixed:
                eps_v = self._fixed[v].to(device=self.device, dtype=self.dtype)

            y = mech(parents_feat, eps=eps_v)
            xs[v] = y

        return xs