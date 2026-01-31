from typing import Any, Dict, Optional, List
import torch
from torch import nn, Tensor


class SimpleMechanism(nn.Module):
    """
    A simple mechanism consisting of a single linear layer followed by an activation.

    Constructor Parameters
    ----------------------
    node_names : List[str]
        The names of all nodes in the SCM.
    activation : nn.Module
        The activation function to apply after the weighted sum.
    device : torch.device
        The device on which to place the parameters.
    generator : torch.Generator, optional
        RNG for reproducibility of activation sampling.
    """

    def __init__(self, node_names: List[str], activation: nn.Module, device: torch.device, generator: Optional[torch.Generator] = None) -> None:
        super().__init__()
        self.generator = generator
        self.activation = activation
        self.device = device
        weights_map = {}
        for v in node_names:
            initial_value = 2 * torch.rand(1, device=self.device, generator=self.generator) - 1
            weights_map[v] = nn.Parameter(initial_value)
        self.weights = nn.ParameterDict(weights_map)
        bias_value = 2 * torch.rand(1, device=self.device, generator=self.generator) - 1
        self.bias = nn.Parameter(bias_value)
        
    def __str__(self) -> str:
        info = []
        info.append(f"Activation: {self.activation}")
        info.append(f"Bias: {self.bias.item()}")
        weights = {v: value.item() for v, value in self.weights.items()}
        info.append(f"Weights: {weights}")
        return "\n".join(info)

    def forward(self, parent_values: Dict[Any, Tensor], eps: Tensor) -> Tensor:  
        if len(parent_values) == 0:
            return eps
        
        weighted_inputs = []
        for v, weight in self.weights.items():
            if v in parent_values:
                weighted_inputs.append(parent_values[v] * weight)
        combined = torch.sum(torch.stack(weighted_inputs), dim=0)
        return self.activation(combined + self.bias) + eps