from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import xgboost as xgb
from prior.mechanisms.base_mechanism import BaseMechanism


class XGBoostLayer(torch.nn.Module):
    """
    A single XGBoost layer that mimics a neural network layer.
    
    This layer fits an XGBoost multi-output regressor on random data 
    during initialization and then uses it for prediction.
    
    Constructor Parameters
    ----------------------
    input_dim : int
        Number of parent features D (can be 0).
    output_dim : int
        Number of output features.
    n_estimators : int
        Number of trees used.
    max_depth : int
        Maximum depth of the trees used.
    generator : torch.Generator, optional
        RNG for reproducibility of fitting XGBoost models.
    n_training_samples : int
        Number of training samples used to fit the XGBoost models.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        n_estimators: int, 
        max_depth: int,
        generator: Optional[torch.Generator] = None,
        n_training_samples: int = 1000
    ):
        super().__init__()   
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = generator
        
        # Get numpy generator from torch generator
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        np_rng = np.random.default_rng(np_seed)
        
        # Sample training inputs and targets
        X_train = np_rng.standard_normal((n_training_samples, input_dim))
        y_train = np_rng.standard_normal((n_training_samples, output_dim))
        
        # Create and fit XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=np_seed if generator is not None else None,
            n_jobs=1,  # Keep deterministic
            verbosity=0  # Suppress output
        )
        
        # For multi-output, we need to handle each output separately
        # or use MultiOutputRegressor wrapper
        if output_dim == 1:
            self.model.fit(X_train, y_train.ravel())
            self._is_multioutput = False
        else:
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
            self.model.fit(X_train, y_train)
            self._is_multioutput = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the XGBoost layer."""
        # Normalize input
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x =  (x - mean) / std
        
        # Convert tensor to numpy for XGBoost prediction
        x_np = x.detach().cpu().numpy()
        
        # Predict using the fitted model
        # we need to reshape, since the model expects 2-D input
        B, N, D = x_np.shape
        x_np = x_np.reshape(B * N, D)
        y_pred = self.model.predict(x_np)
        y_pred = y_pred.reshape(B, N, self.output_dim) # type: ignore
        
        # Convert back to tensor
        y_pred = torch.from_numpy(y_pred).float().to(x.device)

        return y_pred


class SampleXGBoostMechanism(BaseMechanism):
    """
    Randomly-sampled XGBoost-based mechanism following the tree-based SCM prior.

    This mechanism replaces linear layers and activations with XGBoost models fitted on random data.

    Architecture sampling:
    - n_estimators ~ min{4, 1 + Exponential(λ=0.5)}
    - max_depth ~ min{4, 2 + Exponential(λ=0.5)}

    Constructor Parameters
    ----------------------
    input_dim : int
        Number of parent features D (can be 0).
    node_shape : int
        Output per-sample dimension.
    num_hidden_layers : int, default 0
        Number of hidden XGBoost layers.
    hidden_dim : int, default 64
        Width of hidden layers (number of outputs from each XGBoost layer).
    generator : torch.Generator, optional
        RNG for reproducibility of architecture sampling.
    n_training_samples : int, default 1000
        Number of random training samples to use for fitting each XGBoost model.
    add_noise : bool, default True
        Whether to add noise to the output of the XGBoost layers.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        node_dim: int = 1,
        num_hidden_layers: int = 0,
        hidden_dim: int = 64,
        generator: Optional[torch.Generator] = None,
        n_training_samples: int = 1000,
        add_noise: bool = False
    ) -> None:
        super().__init__(input_dim=input_dim, node_dim=node_dim)
        
        self.gen = generator
        self.n_training_samples = n_training_samples
        self.add_noise = add_noise

        n_hidden = num_hidden_layers
        if n_hidden < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        # Build XGBoost layer stack
        self.xgb_layers = torch.nn.ModuleList()

        if n_hidden == 0:
            # Direct mapping from input to output
            n_est, max_d = self._sample_xgboost_params()
            layer = XGBoostLayer(
                input_dim=input_dim,
                output_dim=node_dim,
                n_estimators=n_est,
                max_depth=max_d,
                generator=self.gen,
                n_training_samples=self.n_training_samples
            )
            self.xgb_layers.append(layer)
        else:
            # Stack of hidden layers + output layer
            current_dim = input_dim

            # Hidden layers
            for _ in range(n_hidden):
                n_est, max_d = self._sample_xgboost_params()
                layer = XGBoostLayer(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    n_estimators=n_est,
                    max_depth=max_d,
                    generator=self.gen,
                    n_training_samples=self.n_training_samples
                )
                self.xgb_layers.append(layer)
                current_dim = hidden_dim

            # Output layer
            n_est, max_d = self._sample_xgboost_params()
            output_layer = XGBoostLayer(
                input_dim=current_dim,
                output_dim=node_dim,
                n_estimators=n_est,
                max_depth=max_d,
                generator=self.gen,
                n_training_samples=self.n_training_samples
            )
            self.xgb_layers.append(output_layer)

        # final layer, used after adding noise
        n_est, max_d = self._sample_xgboost_params()
        self.post_xgb_layer = XGBoostLayer(
            input_dim=node_dim,
            output_dim=node_dim,
            n_estimators=n_est,
            max_depth=max_d,
            generator=self.gen,
            n_training_samples=self.n_training_samples
        )

    def _sample_xgboost_params(self) -> Tuple[int, int]:
        """
        Sample XGBoost hyperparameters according to the specification:
        - n_estimators ~ min{4, 1 + Exponential(λ=0.5)}
        - max_depth ~ min{4, 2 + Exponential(λ=0.5)}
        """
        # Sample using generator
        # (can't use exponential distribution since that doesn't support generators)
        exp_sample_1 = torch.empty(1).exponential_(0.5, generator=self.gen).item()
        exp_sample_2 = torch.empty(1).exponential_(0.5, generator=self.gen).item()

        n_estimators = min(4, int(1 + exp_sample_1))
        max_depth = min(4, int(2 + exp_sample_2))

        # Ensure minimum values
        n_estimators = max(1, n_estimators)
        max_depth = max(1, max_depth)

        return n_estimators, max_depth

    def _forward(self, parents: Tensor, eps: Tensor) -> Tensor:
        """Forward pass through the XGBoost mechanism."""
        B = parents.shape[0]
        x = parents
        
        # Forward through XGBoost layers
        for layer in self.xgb_layers:
            x = layer(x)
        out = x

        if not self.add_noise:
            eps = torch.zeros_like(out)

        # Apply final XGBoost transformation after adding noise
        out = out + eps
        out = self.post_xgb_layer(out)
        return out