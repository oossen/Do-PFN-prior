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
            info.append(f"Activation: {self.mechanisms[v].activation}")
            info.append(f"Bias: {self.mechanisms[v].bias}")
            info.append(f"Weights: {self.mechanisms[v].weights}")
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

        # only return data for non-hidden nodes
        xs = {v: x for v, x in xs.items() if not self.dag.nodes[v].get("hidden", False)}
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
        y_values : Tensor
            The values of the target variable `y_var` for which to compute the log-likelihood.
            Must be of shape (B,).
        y_var : str
            The name of the target variable.
            
        Returns
        -------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            If the entries of `values` have shape (*), the returned tensor has shape (*, B).
        """
        y_values = y_values.to(device=self.device, dtype=self.dtype)    
        shape_y = y_values.shape
        assert len(shape_y) == 1, f"y_values must be of shape (B,), but found {shape_y}"
        values_shape = list(values.values())[0].shape
        output_shape = (*values_shape, shape_y[0])
        values_tensor = {y_var: y_values.expand(output_shape)}
        for v, value in values.items():
            if v != y_var:
                values_tensor[v] = value.unsqueeze(-1).expand(output_shape)
                
        log_prob = self.total_log_probability(values_tensor)
        # shift before integration for numerical stability
        max_log_prob = torch.max(log_prob, dim=-1, keepdim=True)[0]
        relative_prob = torch.exp(log_prob - max_log_prob)
        marginal_relative = torch.trapezoid(relative_prob, y_values, dim=-1)
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
    def marginal(self, values: Dict[str, float], steps=100, low=-10.0, high=10.0) -> float:
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