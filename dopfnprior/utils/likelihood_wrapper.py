import torch
from torch import Tensor
from typing import Dict, Optional, Callable, List
import matplotlib.pyplot as plt
import os
import networkx as nx


class LikelihoodWrapper:
    """
    This is a wrapper class for computing the log-likelihood of a target variable given the other variables in an SCM.
    We assume that the SCM in question is an additive noise model with structural equations of the following form:
    
        x_v = f_v(parents_v) + eps_v
        
    Here we use the following notation:
        v : a node in the DAG underlying the SCM. We assume that the v are indexed by strings.
        x_v : the value sampled at node v.
        f_v : the *deterministic* mechanism function of node v.
        parents_v : the values of the parents of v, that is, {x_w | w -> v}
        eps_v : the additive noise term sampled at node v.
        
    We assume that each x_v and each eps_v is a torch.Tensor of shape
    
        (S, dim_v)
        
    where dim_v refers to the node dimension of node v, and where S is a fixed shape common to all nodes.
    For example, we might have S = (batch_size, n_rows).
    
    Parameters
    ----------
    dag : nx.DiGraph
        The graph underlying the SCM.
        It should be a directed acyclic graph (DAG) with string nodes v.
    mechanisms : Dict[str, Callable]
        A dictionary indexed by the nodes v containing the mechanisms f_v.
        Each f_v should be a function which takes as input a dictionary parents_v of parent values x_w and outputs the tensor f_v(parents_v).
        If input is of shape (S, dim_v), output should depend only on the dim_v dimension, not on the dimensions contained in S.
    noise_dists : Dict[str, Callable]
        A dictionary specifying log-likelihoods for the noise distributions at each node.
        noise_dists[v](eps_v) should return the log-likelihood of sampling eps_v.
        If input is of shape (S, dim_v), output should be of shape S, so the probability is provided for all features contained in node v jointly.
        Output should depend only on the dim_v dimension, not on the dimensions contained in S.
    """
    
    
    def __init__(self, 
                 dag: nx.DiGraph, 
                 mechanisms: Dict[str, Callable], 
                 noise_dists: Dict[str, Callable]):
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise_dists = noise_dists
        self._topo: List[str] = list(nx.topological_sort(dag))
        self._parents = {v: list(self.dag.predecessors(v)) for v in self._topo}

    def total_log_probability(self, values: Dict[str, Tensor]) -> Tensor:
        """
        Compute the probabilities of the provided values under the SCM.
        
        Input
        -----
        values : Dict[str, Tensor]
            Dictionary of sampled values.
            The keys should range over the nodes of `self.dag`.
            The value at node v should be a tensor of shape (S, dim_v), where S is a fixed shape.
            
        Output
        ------
        log_prob : Tensor
            The log probability of the provided values under the SCM.
            Has shape (S,), where S is the fixed shape of the input tensors.

        """
        log_probs = []
        sampled_noise = {}
        for v in self._topo:
            sampled_noise[v] = values[v]
            parents_feat = {p: values[p] for p in self._parents[v]}
            if len(parents_feat) > 0:
                mech = self.mechanisms[v]
                x = mech(parents_feat)
                sampled_noise[v] = sampled_noise[v] - x
            log_probs.append(self.noise_dists[v](sampled_noise[v]))
        return torch.stack(log_probs).sum(dim=0)

    @torch.no_grad()
    def log_likelihood_batch(self,
                            values: Dict[str, Tensor], 
                            y_values: Tensor, 
                            y_var: str = 'y',
                            y_idx: int = 0,
                            plot_dir: Optional[str] = None) -> Tensor:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        
        Input
        -----
        values : Dict[str, Tensor]
            Dictionary of sampled values.
            If `y_var` is included, its values are ignored.
            Each value should have shape (batch_size, n_rows, dim_v).
        y_values : Tensor
            The values of feature `y_idx`of the target variable `y_var` for which to compute the log-likelihood.
            Should have shape (batch_size, n_y_values).
        y_var : str
            The name of the target node in the DAG.
        y_idx : int
            The index of the target feature in the node `y_var`.
        plot_dir : str, optional
            If not None, the directory where to save likelihood plots.
            
        Output
        ------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            Has shape (batch_size, n_rows, n_y_values).
        """ 
        device = list(values.values())[0].device
        y_values = y_values.to(device)
        batch_size, n_y_values = y_values.shape
        n_rows = list(values.values())[0].shape[1]
        # construct the tensor of all possible combinations of values with the provided y_values
        # we replace only the feature y_idx of y_var with the provided y_values
        value_tensors = {}
        y_shape_left = (batch_size, n_rows, n_y_values, y_idx)
        y_shape_right = (batch_size, n_rows, n_y_values, values[y_var].shape[2] - y_idx - 1)
        replacement_y = y_values.unsqueeze(1).expand((batch_size, n_rows, n_y_values)).unsqueeze(-1)
        left_y = values[y_var][:, :, :y_idx].unsqueeze(2).expand(y_shape_left)
        right_y = values[y_var][:, :, y_idx+1:].unsqueeze(2).expand(y_shape_right)
        value_tensors[y_var] = torch.cat((left_y, replacement_y, right_y), dim=-1)
        for v, value in values.items():
            if v != y_var:
                value_shape = (batch_size, n_rows, n_y_values, value.shape[2])
                value_tensors[v] = value.unsqueeze(2).expand(value_shape)
                
        log_prob = self.total_log_probability(value_tensors)
        # shift before integration for numerical stability
        max_log_prob = torch.max(log_prob, dim=-1, keepdim=True)[0]
        relative_prob = torch.exp(log_prob - max_log_prob)
        marginal_relative = torch.trapezoid(relative_prob, y_values.unsqueeze(1).expand((batch_size, n_rows, n_y_values)), dim=-1)
        log_marginal = torch.log(marginal_relative).unsqueeze(-1) + max_log_prob
        
        ret = log_prob - log_marginal
        
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            # only plot the first batch
            for i in range(n_rows):
                plt.figure()
                probs = torch.exp(ret[0, i, :])
                plt.plot(y_values[0].cpu().numpy(), probs.cpu().numpy())
                true_y = values[y_var][0, i, y_idx].cpu().item()
                plt.axvline(true_y, color='red', linestyle='--', label='True value')
                plt.legend()
                plt.xlabel(y_var)
                plt.ylabel(f"p({y_var})")
                # set xlim to exclude values very close to 0
                eps = 0.01 * probs.max()
                mask = probs > eps
                indices = torch.where(mask)[0]
                buffer = 1
                start_idx = max(0, indices[0] - buffer)
                end_idx = min(len(y_values[0]) - 1, indices[-1] + buffer)
                plt.xlim(y_values[0][start_idx].cpu().item(), y_values[0][end_idx].cpu().item())
                plt.savefig(f"{plot_dir}/likelihood_row_{i}.png")
                plt.close()

        return ret