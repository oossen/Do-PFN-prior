from typing import Optional, Tuple

import torch
import torch.nn as nn


class StdScaleLayer(nn.Module):
    """Standard scaling layer that normalizes input features.

    Computes mean and standard deviation on the first batch and uses these
    statistics to normalize subsequent inputs using (x - mean) / std.
    The statistics are computed along dimension 1, the data sample dimension.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fit the info on the first batch
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + 1e-6

        return (x - self.mean) / self.std


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class StdRandomScaleFactory:
    def __init__(self, act_class, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(StdScaleLayer(), self.act_class())


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