from typing import Dict, Optional, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import networkx as nx

from dopfnprior.utils.sampling import DistributionSampler


class SCM:
    """
    Structural Causal Model with vectorized ancestral sampling.

    Workflow
    --------
    1) scm.sample_noise(B)                 # samples and fixes noise (B is any shape)
    2) xs = scm.propagate()                # uses the fixed noises

    Parameters
    ----------
    dag : CausalDAG
    mechanisms : Dict[str, nn.Module]
    noise : Dict[str, DistributionSampler]
    device : torch.device
    dtype : torch.dtype
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        mechanisms: Dict[str, nn.Module],
        noise: Dict[str, DistributionSampler],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = device
        self.dtype = dtype

        # Topology and parents
        self._topo: List[str] = list(nx.topological_sort(dag))
        self._parents: Dict[str, List[str]] = {v: list(self.dag.predecessors(v)) for v in self._topo}

        # Fixed noise buffers
        self._sampled_noise: Dict[str, Tensor] = {}
        
    def __str__(self) -> str:
        info = []
        info.append(f"SCM on device={self.device}, dtype={self.dtype}")
        info.append(f"Nodes: {list(self.dag.nodes())}")
        for v in self._topo:
            info.append(v)
            info.append(f"Parents: {self._parents[v]}")
            info.append(f"Mechanism: {self.mechanisms[v]}")
            info.append(f"Noise std: {self.noise[v].std()}")
        return "\n".join(info)
    
    @torch.no_grad()
    def sample_noise(self, sample_shape: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> None:
        for v in self._topo:
            dist_v = self.noise[v]
            eps_v = dist_v.sample_shape(sample_shape, generator=generator)
            self._sampled_noise[v] = eps_v

    @torch.no_grad()
    def propagate(self) -> Dict[str, Tensor]:
        xs: Dict[str, Tensor] = {}
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {}
            for p in self._parents[v]:
                parents_feat[p] = xs[p]
            eps_v = self._sampled_noise[v].to(device=self.device, dtype=self.dtype)

            x = mech(parents_feat, eps=eps_v)
            xs[v] = x

        return xs
    
    @torch.no_grad()
    def log_likelihood_batch(self, values: Dict[str, Tensor], y_values: Tensor, y_var: str = 'y') -> Tensor:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        
        Parameters
        ----------
        values : Dict[str, Tensor]
            Contains observed values for each feature.
            If `y_var` is included, its values are ignored.
            Each value should have shape (batch_size, n_cols).
        y_values : Tensor
            The values of the target variable `y_var` for which to compute the log-likelihood.
            Should have shape (batch_size, n_y_values).
        y_var : str
            The name of the target variable.
            
        Returns
        -------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            Has shape (batch_size, n_cols, n_y_values).
        """
        y_values = y_values.to(device=self.device, dtype=self.dtype)    
        batch_size, n_y_values = y_values.shape
        values_shape = list(values.values())[0].shape
        assert values_shape[0] == batch_size, "Batch size of values and y_values must match."
        n_cols = values_shape[1]
        output_shape = (batch_size, n_cols, n_y_values)
        values_tensor = {y_var: y_values.unsqueeze(1).expand(output_shape)}
        for v, value in values.items():
            if v != y_var:
                values_tensor[v] = value.unsqueeze(-1).expand(output_shape)
                
        log_prob = self.total_log_probability(values_tensor)
        # shift before integration for numerical stability
        max_log_prob = torch.max(log_prob, dim=-1, keepdim=True)[0]
        relative_prob = torch.exp(log_prob - max_log_prob)
        marginal_relative = torch.trapezoid(relative_prob, values_tensor[y_var], dim=-1)
        log_marginal = torch.log(marginal_relative).unsqueeze(-1) + max_log_prob

        return log_prob - log_marginal
    
    @torch.no_grad()
    def total_log_probability(self, values: Dict[str, Tensor]) -> Tensor:
        """
        Compute the probabilities of the provided values under the SCM.
        The returned tensor has the same shape as the input values.
        """
        sampled_noise = {}
        value_shape = list(values.values())[0].shape
        values = {v: value.to(device=self.device, dtype=self.dtype) for v, value in values.items()}
        log_prob = torch.zeros(value_shape, device=self.device, dtype=self.dtype)
        for v in self._topo:
            mech = self.mechanisms[v]
            parents_feat = {p: values[p] for p in self._parents[v]}
            x = mech(parents_feat, eps=torch.zeros(value_shape, device=self.device, dtype=self.dtype))
            sampled_noise[v] = values[v] - x
            log_prob += self.noise[v].log_prob(sampled_noise[v])
        return log_prob
    
    @torch.no_grad()
    def marginal(self, values: Dict[str, Tensor], steps=100, low=-10.0, high=10.0) -> Tensor:
        """
        Compute the marginal probabilities of the provided values.
        Currently assumes that exactly one variable is marginalized out.
        The returned tensor has the same shape as the input values.
        """
        expected_keys = self.dag.nodes()
        missing = set(expected_keys) - set(values.keys())
        assert len(missing) == 1, f"Expected exactly 1 missing key, but found {len(missing)}: {missing}"
        
        y_values = torch.linspace(low, high, steps, device=self.device, dtype=self.dtype)
        y_var = missing.pop()
        values_shape = list(values.values())[0].shape
        output_shape = values_shape + (steps,)
        values_tensor = {y_var: y_values.unsqueeze(0).unsqueeze(0).expand(output_shape)}
        for v, value in values.items():
            if v != y_var:
                values_tensor[v] = value.unsqueeze(-1).expand(output_shape)
        log_prob = self.total_log_probability(values_tensor)
        max_log_prob = torch.max(log_prob, dim=-1, keepdim=False)[0]
        relative_prob = torch.exp(log_prob - max_log_prob)
        marginal_relative = torch.trapezoid(relative_prob, values_tensor[y_var], dim=-1)
        log_marginal = torch.log(marginal_relative) + max_log_prob
        
        return log_marginal