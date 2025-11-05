from typing import List, Optional
import torch
from torch import nn, Tensor
from mechanisms.base_mechanism import BaseMechanism
from mechanisms.tab_icl_activations import RandomActivation


class SampleMLPMechanism(BaseMechanism):
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
        self.gen = generator

        # use fixed number of hidden layers
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        n_hidden = num_hidden_layers

        layers: List[nn.Module] = []
        if input_dim == 0:
            # no model needed
            self.net = None
        else:
            if n_hidden == 0:
                layers.append(_deterministic_linear_layer(input_dim, node_dim, generator=self.gen))
            else:
                d = input_dim
                act = RandomActivation(generator=self.gen)
                for _ in range(n_hidden):
                    layers += [_deterministic_linear_layer(d, hidden_dim, generator=self.gen), act]
                    d = hidden_dim
                layers.append(_deterministic_linear_layer(d, node_dim, generator=self.gen))
            self.net = nn.Sequential(*layers)

        # final layer, used after adding noise (the only activation if there are no hidden layers)
        post_linear = _deterministic_linear_layer(node_dim, node_dim, generator=self.gen)
        post_activation = RandomActivation(generator=self.gen)
        self.post_layer = nn.Sequential(post_linear, post_activation)

    def _forward(self, parents: Tensor, eps: Tensor) -> Tensor:
        if self.net is None:
            B, N, D = parents.shape
            out = torch.zeros((B, N, self.node_dim), device=parents.device, dtype=parents.dtype)
        else:
            out = self.net(parents)      
        out = out + eps
        out = self.post_layer(out)
        return out 
    

def _deterministic_linear_layer(input_dim: int, output_dim: int, generator: Optional[torch.Generator]):
    """Return a newly initialized linear layer with the sampling of the weight controlled by `generator`."""
    bound = 1 / input_dim**0.5
    layer = nn.Linear(input_dim, output_dim, bias=False)
    nn.init.uniform_(layer.weight, -bound, bound, generator=generator)
    return layer