import math
from typing import List, Optional
import torch
from torch import nn, Tensor

from dopfnprior.mechanisms.base_mechanism import BaseMechanism
from dopfnprior.mechanisms.activations import RandomActivation


class MLPMechanism(BaseMechanism):
    """
    Randomly-sampled MLP mechanism with a fixed (sampled) activation module.

    Constructor Parameters
    ----------------------
    input_dim : int
        Number of parent features D (can be 0).
    node_dim : int
        Output per-sample dimension.
    num_hidden_layers : int
        Fixed number of hidden layers.
    hidden_dim : int, default 64
        Width of hidden layers.
    generator : torch.Generator, optional
        RNG for reproducibility of activation sampling.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        node_dim: int = 1,
        num_hidden_layers: int = 2,
        hidden_dim: int = 64,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(input_dim=input_dim, node_dim=node_dim)
        self.generator = generator

        # use fixed number of hidden layers
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        n_hidden = num_hidden_layers

        layers: List[nn.Module] = []
        if input_dim == 0:
            # no model needed
            self.net = None
        else:
            d = input_dim
            for _ in range(n_hidden):
                act = RandomActivation(generator=self.generator)
                linear_layer = _deterministic_linear_layer(d, hidden_dim, generator=self.generator)
                layers += [linear_layer, act]
                d = hidden_dim
            act = RandomActivation(generator=self.generator)
            linear_layer = _deterministic_linear_layer(d, node_dim, generator=self.generator)
            layers += [linear_layer, act]
            self.net = nn.Sequential(*layers)

    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        out_shape = parents.shape[:-1] + (self.node_dim,)
        if self.net is None:
            out = torch.zeros(out_shape, device=parents.device, dtype=parents.dtype)
        else:
            out = self.net(parents)
        if eps is not None:      
            out = out + eps
        return out
    
    def log_prob(self) -> float:
        total_log_prob = 0.0
        if self.net is None:
            return total_log_prob
        for layer in self.net:
            if isinstance(layer, RandomActivation):
                total_log_prob += layer.log_prob()
            elif isinstance(layer, nn.Linear):
                bound = 1 / layer.in_features**0.5
                total_log_prob += -layer.weight.numel() * math.log(1.0 / (2 * bound))
        return total_log_prob
    

def _deterministic_linear_layer(input_dim: int, output_dim: int, generator: Optional[torch.Generator]):
    """Return a newly initialized linear layer with the sampling of the weight controlled by `generator`."""
    bound = 1 / input_dim**0.5
    layer = nn.Linear(input_dim, output_dim, bias=False)
    nn.init.uniform_(layer.weight, -bound, bound, generator=generator)
    return layer