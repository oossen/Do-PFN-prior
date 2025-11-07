from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Type
import math
import torch
from torch import Tensor
import torch.distributions as dist

from dopfnprior.utils.hyperparameter_sampling import TorchDistributionSampler


class MixedDist:
    """
    Scalar mixture distribution for additive noise. Each scalar draw:
      1) samples a component i ~ Categorical(mixture_proportions)
      2) samples x ~ component_i  (with zero mean; finite variance)

    Components supported by default: Normal, Laplace, StudentT(df=3), Gumbel (mean-shifted to 0).
    The 'std' parameter is converted to each component's scale via 'std2scale'.
    """

    def __init__(
        self,
        std: float,
        distributions: Sequence[Type[dist.Distribution]] = (dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        std2scale: Optional[dict[Type[dist.Distribution], Callable[[float], float]]] = None,
        student_t_df: float = 3.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Parameters
        ----------
        std : float
            Target standard deviation (common across components; converted to scale per component).
        distributions : sequence of torch.distributions classes
            Component distribution classes (subset of Normal, Laplace, StudentT, Gumbel).
        mixture_proportions : sequence of floats
            Nonnegative weights; normalized internally to sum to 1.
        std2scale : optional dict
            Mapping {DistributionClass: std -> scale}. If None, sensible defaults are used.
        student_t_df : float
            Degrees of freedom for StudentT; default 3 ensures finite variance.
        """
        self.device = torch.device(device)
        self.dtype = dtype

        if len(distributions) != len(mixture_proportions):
            raise ValueError("distributions and mixture_proportions must have the same length.")
        if any(w < 0 for w in mixture_proportions):
            raise ValueError("mixture_proportions must be nonnegative.")

        self.std = float(std)
        self.distributions = list(distributions)
        self.student_t_df = float(student_t_df)

        # Normalize proportions
        w = torch.as_tensor(mixture_proportions, dtype=self.dtype, device=self.device)
        w_sum = float(w.sum().item())
        if w_sum <= 0:
            raise ValueError("mixture_proportions must sum to a positive value.")
        self.weights = (w / w_sum)

        # Default std -> scale converters
        if std2scale is None:
            # For StudentT with df>2, Var = df/(df-2) * scale^2  => scale = std*sqrt((df-2)/df)
            # For Gumbel, Var = (pi^2/6) * scale^2  => scale = std * sqrt(6)/pi, mean=loc + gamma*scale
            std2scale = {
                dist.Normal:  lambda s: s,
                dist.Laplace: lambda s: s / math.sqrt(2.0),
                dist.StudentT: lambda s, df=self.student_t_df: s * math.sqrt((df - 2.0) / df),
                dist.Gumbel:  lambda s: s * math.sqrt(6.0) / math.pi,
            }

        self.std2scale = std2scale
        self._components = self._init_components()  # list of instantiated torch distributions
        cat_dist = dist.Categorical(probs=self.weights)
        self._cat = TorchDistributionSampler(cat_dist) # wrap the distribution to support generators

    def _init_components(self) -> List[TorchDistributionSampler]:
        comps: List[TorchDistributionSampler] = []
        euler_gamma = 0.5772156649015329

        for cls in self.distributions:
            if cls not in self.std2scale:
                raise ValueError(f"Missing std2scale mapping for {cls.__name__}.")

            if cls is dist.StudentT:
                scale = torch.tensor(self.std2scale[cls](self.std), device=self.device, dtype=self.dtype)
                df = torch.tensor(self.student_t_df, device=self.device, dtype=self.dtype)
                loc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                comps.append(TorchDistributionSampler(dist.StudentT(df=df, loc=loc.item(), scale=scale.item())))

            elif cls is dist.Gumbel:
                scale = torch.tensor(self.std2scale[cls](self.std), device=self.device, dtype=self.dtype)
                # set mean to zero: mean = loc + gamma*scale => loc = -gamma*scale
                loc = -torch.tensor(euler_gamma, device=self.device, dtype=self.dtype) * scale
                comps.append(TorchDistributionSampler(dist.Gumbel(loc=loc, scale=scale)))

            elif cls is dist.Normal:
                scale = torch.tensor(self.std2scale[cls](self.std), device=self.device, dtype=self.dtype)
                loc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                comps.append(TorchDistributionSampler(dist.Normal(loc=loc, scale=scale)))

            elif cls is dist.Laplace:
                scale = torch.tensor(self.std2scale[cls](self.std), device=self.device, dtype=self.dtype)
                loc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                comps.append(TorchDistributionSampler(dist.Laplace(loc=loc, scale=scale)))

            else:
                raise ValueError(f"Unsupported distribution class: {cls.__name__}")

        return comps

    @torch.no_grad()
    def sample_n(self, n: int, generator: torch.Generator) -> Tensor:
        """
        Vectorized i.i.d. sampling: draw component indices for N scalars,
        then sample from all components and select.
        Returns shape (n,).
        """
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)

        idx = self._cat.sample_n(n, generator=generator)  # (n,)
        out = torch.empty((n,), device=self.device, dtype=self.dtype)

        # Sample from each component only where needed
        for k, comp in enumerate(self._components):
            mask = (idx == k)
            count = int(mask.sum().item())
            if count > 0:
                out[mask] = comp.sample_n(count, generator).to(self.device, self.dtype)

        return out

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...], generator: torch.Generator) -> Tensor:
        """
        Fully vectorized sampling for any output shape.
        """
        N = int(math.prod(shape))
        flat = self.sample_n(N, generator=generator)
        return flat.reshape(shape)