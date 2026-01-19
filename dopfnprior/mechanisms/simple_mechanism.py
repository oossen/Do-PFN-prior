from typing import Any, Dict, Optional
import torch
from torch import nn, Tensor
from dopfnprior.mechanisms.activations import RandomActivation


class SimpleMechanism(nn.Module):
    """
    A simple mechanism consisting of a single linear layer followed by a random activation.

    Constructor Parameters
    ----------------------
    node_names : List[str]
        The names of all nodes in the SCM.
    generator : torch.Generator, optional
        RNG for reproducibility of activation sampling.
    """

    def __init__(self, node_names: list, generator: Optional[torch.Generator] = None) -> None:
        super().__init__()
        self.generator = generator
        weights_map = {}
        for v in node_names:
            initial_value = 2 * torch.rand(1, generator=self.generator) - 1
            weights_map[v] = nn.Parameter(initial_value)
        self.weights = nn.ParameterDict(weights_map)

    def forward(self, parent_values: Dict[Any, Tensor], eps: Tensor) -> Tensor:  
        if len(parent_values) == 0:
            return eps
        
        weighted_inputs = []
        for v, weight in self.weights.items():
            if v in parent_values:
                weighted_inputs.append(parent_values[v] * weight)
        combined = torch.sum(torch.stack(weighted_inputs), dim=0)
        return torch.asinh(combined) + eps