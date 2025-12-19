from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import torch
import torch.distributions as dist


class DistributionSampler(ABC):
    """Abstract base class for distribution samplers."""
    
    @abstractmethod
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample n values from this distribution."""
        pass
    
    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> float:
        """
        Compute log probability of the given value(s).
        The output is float valued even if `value` is not a singleton. 
        """
        pass
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        """Sample just one value."""
        singleton_tensor = self.sample_n(1, generator)
        return singleton_tensor.item()
    
    def sample_shape(self, shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        """Fully vectorized sampling for any output shape."""
        N = int(math.prod(shape))
        flat = self.sample_n(N, generator=generator)
        return flat.reshape(shape)


class FixedSampler(DistributionSampler):
    """Sampler that always returns a fixed value."""
    
    def __init__(self, value: Any):
        self.value = value
    
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.full((n,), self.value)
    
    def log_prob(self, value: torch.Tensor) -> float:
        if torch.all(value == self.value):
            return 0.0  # log(1)
        else:
            return float('-inf')  # log(0)
    

class TorchDistributionSampler(DistributionSampler):
    """
    Wrapper for torch.distributions samplers.
    The important part is adding support for the generator argument.
    """
    
    def __init__(self, distribution: dist.Distribution):
        self.distribution = distribution
        
    def log_prob(self, value: torch.Tensor) -> float:
        log_probs = self.distribution.log_prob(value)
        return log_probs.sum().item()
    
    @torch.no_grad()
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            # Use the generator for sampling
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample((n,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = self.distribution.sample((n,))

        return value


class DiscreteUniformSampler(DistributionSampler):
    """Discrete uniform distribution sampler (integers) using torch."""
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        
    def log_prob(self, value: torch.Tensor) -> float:
        in_range = (value >= self.low) & (value <= self.high)
        num_values = self.high - self.low + 1
        log_prob_value = math.log(1.0 / num_values)
        log_probs = torch.where(in_range, torch.full_like(value, log_prob_value, dtype=torch.float32), torch.full_like(value, float('-inf'), dtype=torch.float32))
        return log_probs.sum().item()
    
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                values = torch.randint(self.low, self.high + 1, (n,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            values = torch.randint(self.low, self.high + 1, (n,))
        
        return values
    

class LogarithmicSampler(DistributionSampler):
    """
    Sample from the interval [low, high] in such a way that
    log of sample is chosen uniformly from [log(low), log(high)].
    """
    def __init__(self, low: float, high: float):
        self.log_low = math.log(low)
        self.log_high = math.log(high)
        self.uniform_sampler = TorchDistributionSampler(dist.Uniform(low=self.log_low, high=self.log_high))
        
    def log_prob(self, value: torch.Tensor) -> float:
        # density of log-uniform distribution is 1/(x * (log(b) - log(a))) for x in [a, b]
        log_probs = -torch.log(value) - math.log(self.log_high - self.log_low)
        return log_probs.sum().item()
    
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        log_sample = self.uniform_sampler.sample_n(n, generator)
        return torch.exp(log_sample)
    

DISTRIBUTION_FACTORIES = {
    "fixed": lambda params: FixedSampler(params["value"]),
    "uniform": lambda params: TorchDistributionSampler(
        dist.Uniform(low=params["low"], high=params["high"])
    ),
    "normal": lambda params: TorchDistributionSampler(
        dist.Normal(loc=params["mean"], scale=params["std"])
    ),
    "lognormal": lambda params: TorchDistributionSampler(
        dist.LogNormal(loc=params["mean"], scale=params["std"])
    ),
    "exponential": lambda params: TorchDistributionSampler(
        dist.Exponential(rate=params["lambd"])
    ),
    "gamma": lambda params: TorchDistributionSampler(
        dist.Gamma(concentration=params["alpha"], rate=params["beta"])
    ),
    "beta": lambda params: TorchDistributionSampler(
        dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
    ),
    "discrete_uniform": lambda params: DiscreteUniformSampler(
        params["low"], params["high"]),
    "logarithmic": lambda params: LogarithmicSampler(
        params["low"], params["high"]
    ),
}


def build_samplers(config: Dict[str, Any],
                   config_name: str,
                   expected_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build sampler objects from the configuration."""
    samplers = {}

    for param_name, param_config in config.items():
        # Check if parameter is known
        if expected_params is not None and param_name not in expected_params:
            raise ValueError(f"Unknown {config_name} hyperparameter: {param_name}")

        # Handle shorthand fixed value notation
        if "value" in param_config and "distribution" not in param_config:
            sampler = FixedSampler(param_config["value"])
                    
        elif "distribution" in param_config:
            dist_type = param_config["distribution"]
            # Regular parameter: create sampler
            if dist_type not in DISTRIBUTION_FACTORIES:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            # Get distribution parameters
            dist_params = param_config.get("distribution_parameters", {})
            if dist_type == "fixed":
                if "value" not in param_config:
                    raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                dist_params = {"value": param_config["value"]}

                # Create sampler
            try:
                sampler = DISTRIBUTION_FACTORIES[dist_type](dist_params)
            except Exception as e:
                raise ValueError(f"Error creating sampler for {config_name}.{param_name}: {e}")
        else:
            raise ValueError(f"Configuration for {config_name}.{param_name} must specify 'distribution' or 'value'")

        samplers[param_name] = sampler

    # Check that all required parameters are specified
    if expected_params is not None:
        required_params = set(expected_params.keys())
        provided_params = set(config.keys())
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required {config_name} parameters: {missing_params}")

    return samplers

@overload
def sample_parameters(samplers: Dict[str, Any], 
                      generator: Optional[torch.Generator]=None, 
                      return_log_prob: Literal[False] = False) -> Dict[str, Any]: ...

@overload
def sample_parameters(samplers: Dict[str, Any], 
                      generator: Optional[torch.Generator]=None, 
                      return_log_prob: Literal[True] = True) -> Tuple[Dict[str, Any], float]: ...

def sample_parameters(samplers: Dict[str, Any], 
                      generator: Optional[torch.Generator]=None, 
                      return_log_prob=False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], float]]:
    """Sample parameters from samplers with type validation."""
    sampled_params = {}
    for param_name, sampler in samplers.items():
        value = sampler.sample(generator)
        sampled_params[param_name] = value
    
    if return_log_prob:
        total_log_prob = 0.0
        for param_name, sampler in samplers.items():
            value = sampled_params[param_name]
            log_prob = sampler.log_prob(torch.tensor([value]))
            total_log_prob += log_prob
        return sampled_params, total_log_prob

    return sampled_params