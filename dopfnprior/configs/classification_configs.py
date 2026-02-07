import torch
import torch.nn as nn


# activation functions
class ArcsinhWrapper(nn.Module):
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
    

prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 800, "high": 1200}
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {
            "value": 100
        },
        # number of classes for the classification problem
        # int
        "num_classes": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 2, "high": 5}
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 5, "high": 120}
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.1, "high": 0.3}
        },
        # probability of making a given node hidden
        # float
        "dropout_prob": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.0, "high": 0.3}
        },
    },

    "noise_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1.0, "shift": 0.1}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 0.1, "shift": 0.1}
        },
    },
    
    "activations": [ArcsinhWrapper(nn.Identity()), 
                    ArcsinhWrapper(nn.LeakyReLU(negative_slope=0.1)), 
                    ArcsinhWrapper(Square()),
                    ArcsinhWrapper(nn.Identity(), swap_sign=True), 
                    ArcsinhWrapper(nn.LeakyReLU(negative_slope=0.1), swap_sign=True), 
                    ArcsinhWrapper(Square(), swap_sign=True)],
}
    

