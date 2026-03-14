from typing import Any, Dict, Optional
import torch
from torch import nn, Tensor


class SimpleMechanism(nn.Module):
    """
    A simple mechanism which linearly combines parent values and then applies an activation function.

    Constructor Parameters
    ----------------------
    node : str
        The name of the node for which this mechanism is defined.
    node_dims : Dict[str, int]
        The dimensions of all nodes in the SCM.
    activation : nn.Module
        The activation function to apply after the weighted sum.
    generator : torch.Generator, optional
        For making the initialization of weights and bias reproducible.
        
    Weights and bias are initialized for every every node in the list of node names provided.
    Thus the forward method should only be called with values of *parent nodes*.
    """

    def __init__(self, 
                 node: str, 
                 node_dims: Dict[str, int], 
                 activation_class: nn.Module, 
                 generator: Optional[torch.Generator] = None) -> None:
        super().__init__()
        self.activation = activation_class(generator=generator)
        linear_layers = {}
        for v in node_dims:
            layer = nn.Linear(node_dims[v], node_dims[node])
            # make initialization of the linear layer deterministic
            nn.init.uniform_(layer.weight, a=-1.0, b=1.0, generator=generator)
            nn.init.uniform_(layer.bias, a=-1.0, b=1.0, generator=generator)
            linear_layers[v] = layer

        self.linear_layers = nn.ModuleDict(linear_layers)

    def forward(self, parent_values: Dict[Any, Tensor]) -> Tensor:  
        contributions = []
        for v, layer in self.linear_layers.items():
            if v in parent_values:
                contributions.append(layer(parent_values[v]))
        combined = torch.stack(contributions).sum(dim=0)
        return self.activation(combined)