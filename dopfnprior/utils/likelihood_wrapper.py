import torch
from torch import Tensor
from typing import Dict


class LikelihoodWrapper:
    """
    This is a wrapper class for computing the log-likelihood of a target variable given the other variables in an SCM.
    The mechanisms and noise distributions of the SCM are abstracted away and replaced by provided functions.
    
    We assume that the values whose log-likelihoods we want to compute come from an additive noise model:
    
        x_v = f_v(parents_v) + eps_v
    
    Parameters
    ----------
    mechanism_fn : Callable
        A function that takes a dictionary of variable values and returns the noise-free values of the variables.
        In the notation above, the value of this dictionary for node v should be f_v(parents_v).
    noise_fn : Callable
        A function that takes a dictionary of noise residuals and returns the log-probability of those residuals under the noise distribution.
        
    Value semantics
    ---------------
    The value of the dictionary `values` processed by `mechanism_fn` can be of shape
        S + (dim_v,),
    where S is any shape of batch dimensions and dim_v is the dimension of node v.
    The output of `mechanism_fn` should have the same shape.
    
    The input to `noise_fn` is a dictionary whose values are also of this same shape.
    The output of `noise_fn` should then consist of values of shape S.
    """
    
    
    def __init__(self, mechanism_fn, noise_fn):
        self.mechanism_fn = mechanism_fn
        self.noise_fn = noise_fn

    @torch.no_grad()
    def total_log_probability(self, values: Dict[str, Tensor]) -> Tensor:
        """
        Compute the probabilities of the provided values.
        The returned tensor has the same shape as the input values.
        """
        noise_free_values = self.mechanism_fn(values)
        noise_residuals = {v: values[v] - noise_free_values[v] for v in values}
        return self.noise_fn(noise_residuals)

    @torch.no_grad()
    def log_likelihood_batch(self,
                            values: Dict[str, Tensor], 
                            y_values: Tensor, 
                            y_var: str = 'y',
                            y_idx: int = 0) -> Tensor:
        """
        Compute the log-likelihood of the provided values of `y_var` conditioned on all other variables.
        
        Parameters
        ----------
        values : Dict[str, Tensor]
            Contains observed values for each feature.
            If `y_var` is included, its values are ignored.
            Each value should have shape (batch_size, n_rows, dim_v).
        mechanism_fn : Callable
            A function that     
        y_values : Tensor
            The values of the target variable `y_var` for which to compute the log-likelihood.
            Should have shape (batch_size, n_y_values).
        y_var : str
            The name of the target node in the DAG.
        y_idx : int
            The index of the target feature in the node `y_var`.
            
        Returns
        -------
        log_likelihood : Tensor
            The log-likelihood of the `y_values` given the other variables in `values`.
            Has shape (batch_size, n_rows, n_y_values).
        """ 
        batch_size, n_y_values = y_values.shape
        n_rows = list(values.values())[0].shape[1]
        value_tensors = {}
        expanded_y_values = y_values.unsqueeze(1).expand((batch_size, n_rows, n_y_values))
        for v, value in values.items():
            value_shape = (batch_size, n_rows, n_y_values, value.shape[2])
            value_tensors[v] = value.unsqueeze(2).expand(value_shape)
        value_tensors[y_var][:, :, :, y_idx] = expanded_y_values
                
        log_prob = self.total_log_probability(value_tensors)
        # shift before integration for numerical stability
        max_log_prob = torch.max(log_prob, dim=-1, keepdim=True)[0]
        relative_prob = torch.exp(log_prob - max_log_prob)
        marginal_relative = torch.trapezoid(relative_prob, expanded_y_values, dim=-1)
        log_marginal = torch.log(marginal_relative).unsqueeze(-1) + max_log_prob

        return log_prob - log_marginal