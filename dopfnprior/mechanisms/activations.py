from typing import Optional, Tuple

import torch
import torch.nn as nn


class StdScaleLayer(nn.Module):
    """
    Perform standard scaling on the input tensor.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        
        out = (x - mean) / std
        return torch.asinh(out)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class StdRandomScaleFactory:
    def __init__(self, act_class, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(self.act_class(), StdScaleLayer())


def get_activations(scale: bool = True):
    "Return the full list of activation functions we use."
    
    # Start with a set of simple activations
    simple_activations = [
        nn.Tanh,
        nn.ReLU,
        SquareActivation,
    ]
    if scale:
        # Create scaled versions using StdRandomScaleFactory
        activations = [StdRandomScaleFactory(act) for act in simple_activations]

    return activations


class RandomActivation(nn.Module):
    """Return a random activation function."""

    def __init__(
        self,
        clamp: Tuple[float, float] = (-1000.0, 1000.0),
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.clamp = clamp
        self.gen = generator
        self._module = self._sample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._module(x)
        if self.clamp is not None:
            y = torch.clamp(y, self.clamp[0], self.clamp[1])
        return y

    def _sample(self) -> nn.Module:
        # Get the the list of activations
        activations = get_activations(scale=True)
            
        # Randomly select one activation function
        idx = int(torch.randint(len(activations), (1,), generator=self.gen).item())
        activation_factory = activations[idx]
            
        # Instantiate the activation
        return activation_factory()