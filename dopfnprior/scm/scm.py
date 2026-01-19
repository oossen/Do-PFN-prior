import math
from typing import Any, Dict, Mapping, Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx
from scipy.integrate import quad
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
        
        """
        # Fit normalization
        n_noise_fitting_samples = 100
        self.sample_noise((n_noise_fitting_samples,), generator=generator)
        self.propagate(generator=generator)
        """
    
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
    def propagate(self, generator: Optional[torch.Generator]) -> Dict[Any, Tensor]:
        xs: Dict[Any, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {}
            for p in self._parents[v]:
                mask = torch.bernoulli(torch.full_like(xs[p], self.dag.edges[(p, v)].get("weight", 1.0)), generator=generator)
                parents_feat[p] = xs[p] * mask
            eps_v = self._sampled_noise[v].to(device=self.device, dtype=self.dtype)

            x = mech(parents_feat, eps=eps_v)
            xs[v] = x

        # only return data for non-hidden nodes
        xs = {v: x for v, x in xs.items() if not self.dag.nodes[v].get("hidden", False)}
        return xs
    
    @torch.no_grad()
    def log_likelihood(self, values: Dict[Any, Tensor], y_var: str = 'y') -> float:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        """
        shape_values = values[list(values.keys())[0]].shape
        total_log_likelihood = 0.0
        for idx in np.ndindex(shape_values):
            values_i = {v: values[v][idx] for v in values}
            joint_log_likelihood = self.total_log_probability(values_i)
            
            def integrand(y):
                values_i[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
                log_prob = self.total_log_probability(values_i)
                return math.exp(log_prob)
            def neg_log_prob(y):
                values_i[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
                log_prob = self.total_log_probability(values_i)
                return -log_prob
            maximum = minimize_scalar(neg_log_prob, bounds=(-20, 20), method='bounded').x # type: ignore
            marginal = quad(integrand, -20, 20, points=[maximum])[0]
            log_marginal = math.log(marginal + EPS)
            
            total_log_likelihood += joint_log_likelihood - log_marginal
        return total_log_likelihood
    
    @torch.no_grad()
    def log_likelihood_batch(self, values: Dict[Any, Tensor], y_values: Tensor, y_var: str = 'y') -> Tensor:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        The output is of the same shape as `y_values`.
        
        Parameters
        ----------
        values : Dict[Any, Tensor]
            Contains observed values for all variables except `y_var`.
            If `y_var` is included, its values are ignored.
        y_values : Tensor
            The values of the target variable `y_var` for which to compute the log-likelihood.
            May have a different shape than the other variables.
        y_var : str
            The name of the target variable.
            
        Returns
        -------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            If `values` contains several samples for each variable, the product of the corresponding
            log-likelihoods is returned for each entry of `y_values`.
        """          
        shape_values = values[list(values.keys())[0]].shape
        shape_y = y_values.shape
        total_log_likelihood = torch.zeros(shape_y, device=self.device, dtype=self.dtype)
        
        for idx in np.ndindex(shape_values):
            values_i = {v: values[v][idx] for v in values}
            
            def integrand(y):
                values_i[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
                log_prob = self.total_log_probability(values_i)
                return math.exp(log_prob)
            log_marginal = log_quad_exp(integrand, -20, 20)
            
            for idx_y in np.ndindex(shape_y):
                values_i[y_var] = y_values[idx_y]
                joint_log_likelihood = self.total_log_probability(values_i)
                total_log_likelihood[idx_y] += joint_log_likelihood - log_marginal

        return total_log_likelihood
    
    @torch.no_grad()
    def total_log_probability(self, values: Dict[Any, Tensor]) -> float:
        """
        Compute the probability of the provided values under the SCM,
        using the mechanisms saved in `self.mechanisms`.
        """
        log_prob = 0.0
        sampled_noise = {}
        value_shape = list(values.values())[0].shape
        for v in self._topo:
            mech = self.mechanisms[v]
            prob_contribution_logs = []
            for r in range(len(self._parents[v]) + 1):
                for parents in combinations(self._parents[v], r):
                    parents_feat = {v: values[p] for p in parents}
                    x = mech(parents_feat, eps=torch.zeros(value_shape, device=self.device, dtype=self.dtype))
                    sampled_noise[v] = values[v] - x
                    prob_contribution_log = 0.0
                    prob_contribution_log += self.noise[v].log_prob(sampled_noise[v])
                    prob_contribution_log += sum(math.log(self.dag.edges[(p, v)].get("weight", 1.0)) for p in parents) \
                        + sum(math.log(1.0 - self.dag.edges[(p, v)].get("weight", 1.0)) for p in self._parents[v] if p not in parents)
                    prob_contribution_logs.append(torch.tensor(prob_contribution_log, device=self.device, dtype=self.dtype))
            all_logs = torch.stack(prob_contribution_logs)
            log_prob += torch.logsumexp(all_logs, dim=0).item()
        return log_prob
    
    @torch.no_grad()
    def marginal(self, values: Dict[Any, Tensor]) -> float:
        """
        Compute the marginal probability of the provided values.
        Currently assumes that exactly one variable is marginalized out.
        """
        expected_keys = self.dag.nodes()
        missing = set(expected_keys) - set(values.keys())
        assert len(missing) == 1, f"Expected exactly 1 missing key, but found {len(missing)}: {missing}"
        y_var = missing.pop()
        def integrand(y):
            values[y_var] = torch.tensor(y, device=self.device, dtype=self.dtype)
            log_prob = self.total_log_probability(values)
            return math.exp(log_prob)
        log_marginal = log_quad_exp(integrand, -20, 20)
        return log_marginal
    

def log_quad_exp(f, a, b):
    """
    Computes log(integral(exp(f(x)) dx)) numerically stably.
    """
    res = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
    max_y, max_val  = res.x, -res.fun # type: ignore
    
    integrand = lambda y: math.exp(f(y) - max_val)
    integral, _ = quad(integrand, a, b, points=[max_y])
    
    return max_val + math.log(integral)