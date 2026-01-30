import torch
import torch.nn as nn


# activation functions
class TanhX2(nn.Module):
    def forward(self, x):
        return torch.tanh(torch.pow(x, 2))
    
class TanhReLU(nn.Module):
    def forward(self, x):
        return torch.tanh(torch.relu(x))
    
class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)
    

prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 100, "high": 300}
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {
            "value": 50
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 5, "high": 10}
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
            "distribution": "exponential",
            "distribution_parameters": {"rate": 1.0}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std_dist": {
            "distribution": "exponential",
            "distribution_parameters": {"rate": 1 / 0.1}
        },
    },
    
    "activations": [TanhX2(), TanhReLU(), Tanh()],
}
    

