from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np


class SignActivation(nn.Module):
    """Sign function as an activation layer.

    Returns 1.0 for inputs >= 0, and -1.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x >= 0.0).float() - 1.0


class Heaviside(nn.Module):
    """Heaviside function as an activation layer.

    Returns 1.0 for inputs >= 0, and 0.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0.0).float()


class RBFActivation(nn.Module):
    """Radial Basis Function (RBF) activation layer.

    Implements the Gaussian RBF: :math:`f(x) = \exp(-x^2)`.
    Useful for localized feature representations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


class RandomFunctionActivation(nn.Module):
    """Random Fourier feature based activation function.

    Generates a random periodic function by combining multiple sine waves with
    different frequencies, phases and weights. The input is first standardized.

    Parameters
    ----------
    n_frequencies : int, default=10
        Number of frequency components to use.
    generator : torch.Generator, optional
        A PyTorch random number generator for reproducibility.
    """

    def __init__(self, generator: torch.Generator, n_frequencies: int = 256):
        super().__init__()

        self.freqs = nn.Parameter(
            n_frequencies * torch.rand(n_frequencies, generator=generator), 
            requires_grad=False
        )
        self.bias = nn.Parameter(
            2 * torch.pi * torch.rand(n_frequencies, generator=generator), 
            requires_grad=False
        )

        with torch.no_grad():
            log_low = torch.log(torch.tensor(1.0))
            log_high = torch.log(torch.tensor(3.0))
            rand_val = torch.rand(1, generator=generator)
            log_sample = log_low + rand_val * (log_high - log_low)
            decay_exponent = -torch.exp(log_sample)

            freq_factors = self.freqs**decay_exponent
            freq_factors = freq_factors / (freq_factors**2).sum().sqrt()
            
        random_normal = torch.randn(n_frequencies, generator=generator)
        self.l2_weights = nn.Parameter(freq_factors * random_normal, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., None] adds a dimension to broadcast against freqs
        x = torch.sin(self.freqs * x[..., None] + self.bias)
        x = (self.l2_weights * x).sum(dim=-1)
        return x


class FunctionActivation(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class ExpActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class SqrtAbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(x))


class UnitIntervalIndicator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.abs(x) <= 1.0).float()


class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class AbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)
    

class AsinhWrapper(nn.Module):
    def __init__(self, activation: nn.Module, swap_sign=False):
        super().__init__()
        self.activation = activation
        self.swap_sign = swap_sign
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        if self.swap_sign:
            x = -x
        return torch.asinh(x)
    
    
class ActivationFactory:
    def __init__(self, act_class):
        self.act_class = act_class

    def __call__(self, generator: Optional[torch.Generator] = None):
        try:
            act = self.act_class(generator=generator)
        except TypeError:
            act = self.act_class()
        swap_sign = torch.rand(1, generator=generator).item() < 0.5
        return AsinhWrapper(act, swap_sign)


def get_activations(random: bool = True):
    """Generate a list of activation functions with various configurations.

    This function creates a list of activation functions by combining simple activations
    with optional random functions, scaling, and diversity options.

    Parameters
    ----------
    random : bool, default=True
        If True, adds RandomFunctionActivation to the list and samples it multiple
        times to increase probability of selection.
    """
    # Start with a set of simple activations
    simple_activations = [
        nn.Tanh,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.SELU,
        nn.SiLU,
        nn.ReLU,
        nn.Softplus,
        nn.ReLU6,
        nn.Hardtanh,
        SignActivation,
        RBFActivation,
        ExpActivation,
        SqrtAbsActivation,
        UnitIntervalIndicator,
        SineActivation,
        SquareActivation,
        AbsActivation,
    ]
    activations = simple_activations
    if random:
        # Add random activation and sample it more often
        activations += [RandomFunctionActivation] * 10

    return [ActivationFactory(act_class) for act_class in activations]