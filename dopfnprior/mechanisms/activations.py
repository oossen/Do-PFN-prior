from typing import Optional, Tuple

import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


def get_activations():
    """
    Return the list of activation functions to be used.
    Add more as needed.
    """
    activations = [
        nn.Tanh,
        nn.ReLU,
        SquareActivation,
    ]
    return activations


class RandomActivation(nn.Module):
    """Randomly activate one of the TabICL activation functions."""

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
        """Sample from TabICL activation functions."""
        # Get the list of activation functions
        activations = get_activations()
            
        # Randomly select one activation function
        idx = int(torch.randint(len(activations), (1,), generator=self.gen).item())
        activation_factory = activations[idx]
            
        # Instantiate the activation
        return activation_factory(generator=self.gen)