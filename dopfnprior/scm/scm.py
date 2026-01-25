import math
from typing import Any, Dict, Mapping, Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar
from itertools import combinations

EPS = 1e-8


class SCM:
    """
    Structural Causal Model with vectorized ancestral sampling.

    Workflow
    --------
    1) scm.sample(B)                       # samples & fixes noise
    2) xs = scm.propagate()                # uses the fixed noises

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
    
    @torch.no_grad()
    def sample_noise(self,
                     sample_shape: Tuple[int, ...],
                     *,
                     generator: Optional[torch.Generator] = None,
                     nodes: Optional[List] = None
                     ) -> Dict[Any, Tensor]:
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
    def propagate(self) -> Dict[Any, Tensor]:
        xs: Dict[Any, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {}
            for p in self._parents[v]:
                parents_feat[p] = xs[p]
            eps_v = self._sampled_noise[v].to(device=self.device, dtype=self.dtype)

            x = mech(parents_feat, eps=eps_v)
            xs[v] = x

        # only return data for non-hidden nodes
        xs = {v: x for v, x in xs.items() if not self.dag.nodes[v].get("hidden", False)}
        return xs
    
    @torch.no_grad()
    def log_likelihood_batch(self, values: Dict[Any, float], y_values: Tensor, y_var: str = 'y') -> Tensor:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        The output is of the same shape as `y_values`.
        
        Parameters
        ----------
        values : Dict[Any, float]
            Contains an observed value for each feature.
            If `y_var` is included, its values are ignored.
        y_values : Tensor
            The values of the target variable `y_var` for which to compute the log-likelihood.
        y_var : str
            The name of the target variable.
            
        Returns
        -------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            Has the same shape as `y_values`.
        """          
        shape_y = y_values.shape
        values_tensor = {y_var: y_values}
        for k, v in values.items():
            if k != y_var:
                values_tensor[k] = torch.full(shape_y, v, device=self.device, dtype=self.dtype)
                
        log_prob = self.total_log_probability(values_tensor)
        prob = torch.exp(log_prob)
        marginal = torch.trapezoid(prob, y_values)
        log_marginal = torch.log(marginal + EPS)

        return log_prob - log_marginal
    
    @torch.no_grad()
    def total_log_probability(self, values: Dict[Any, Tensor]) -> Tensor:
        """
        Compute the probabilities of the provided values under the SCM.
        The returned tensor has the same shape as the input values.
        """
        sampled_noise = {}
        value_shape = list(values.values())[0].shape
        log_prob = torch.zeros(value_shape, device=self.device, dtype=self.dtype)
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {p: values[p] for p in self._parents[v]}
            x = mech(parents_feat, eps=torch.zeros(value_shape, device=self.device, dtype=self.dtype))
            sampled_noise[v] = values[v] - x
            log_prob += self.noise[v].log_prob(sampled_noise[v])
        return log_prob
    
    @torch.no_grad()
    def marginal(self, values: Dict[Any, float], steps=100, low=-10.0, high=10.0) -> float:
        """
        Compute the marginal probability of the provided values.
        Currently assumes that exactly one variable is marginalized out.
        """
        expected_keys = self.dag.nodes()
        missing = set(expected_keys) - set(values.keys())
        assert len(missing) == 1, f"Expected exactly 1 missing key, but found {len(missing)}: {missing}"
        
        y_explore = torch.linspace(low, high, steps, device=self.device, dtype=self.dtype)
        y_var = missing.pop()
        values_tensor = {y_var: y_explore}
        for k, v in values.items():
            if k != y_var:
                values_tensor[k] = torch.full(y_explore.shape, v, device=self.device, dtype=self.dtype)
        
        log_p_explore = self.total_log_probability(values_tensor)
        eps = log_p_explore.max() - 100
        mask = log_p_explore > eps
        indices = torch.where(mask)[0]
        buffer = 1
        start_idx = max(0, indices[0] - buffer)
        end_idx = min(len(y_explore) - 1, indices[-1] + buffer)
        a = y_explore[start_idx]
        b = y_explore[end_idx]

        y = torch.linspace(a, b, steps)
        values_tensor[y_var] = y
        probs = torch.exp(self.total_log_probability(values_tensor))
        marginal = torch.trapezoid(probs, y)
        
        return torch.log(marginal).item()
    

def log_quad_exp(f, a, b) -> float:
    """
    Computes log(integral(exp(f(x)) dx)) numerically stably.
    """
    points = []
    res = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
    max_y, max_val  = res.x, -res.fun # type: ignore
    points.append(max_y)
    
    integrand = lambda y: math.exp(f(y) - max_val)
    integral, _ = quad(integrand, a, b, points=points, epsabs=1e-2, epsrel=1e-2)
    
    return max_val + math.log(integral)