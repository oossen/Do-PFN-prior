from pfns.bar_distribution import get_bucket_limits

import torch
import torch.nn as nn


# activation functions
class AsinhWrapper(nn.Module):
    def __init__(self, activation: nn.Module, swap_sign=False):
        super().__init__()
        self.activation = activation
        self.swap_sign = swap_sign
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        if self.swap_sign:
            x = -x
        return torch.asinh(x)
    
class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)
    
non_linearities = [nn.Identity(), nn.LeakyReLU(negative_slope=0.1), Square()]
activations = [AsinhWrapper(activation) for activation in non_linearities] + [AsinhWrapper(activation, swap_sign=True) for activation in non_linearities]
    

num_outputs = 1000
low, high = -10.0, 10.0
buckets = get_bucket_limits(num_outputs=num_outputs, full_range=(low, high))


prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 50, "high": 100}
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {  # number of test samples per dataset. Can be fixed because architecture is agnostic to the number of test samples.
            "value": 100
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 3, "high": 10}
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.1, "high": 0.4}
        },
        # the number of features contained in each node
        # int
        "features_per_node": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 1, "high": 3},
        },
        # probability of making a given feature hidden
        # float
        "dropout_prob": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.0, "high": 0.3}
        },
    },

    "scm_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 1.0, "shift": 0.2}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 0.3, "shift": 0.1}
        },
        # the activation functions to be used in the SCM
        # categorical distribution over nn.Modules
        "activation_dist": {
            "distribution": "categorical",
            "distribution_parameters": {"choices": activations}
        }
    },
    
}