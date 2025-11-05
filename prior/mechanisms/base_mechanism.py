from abc import ABC, abstractmethod
from torch import nn, Tensor


class BaseMechanism(ABC, nn.Module):
    """
    Minimal parent -> child mechanism for one SCM node.

    Interface
    ---------
    Call: y = mech(parents, eps)

    parents : Tensor, shape (B, N, D)
        Concatenated parent features for the node. D may be 0 (no parents).
    eps : Tensor, shape (B, N, node_dim)
        Node noise.

    Returns
    -------
    Tensor of shape (B, N, node_dim)

    Constructor Parameters
    ----------------------
    input_dim : int
        D — number of parent features the mechanism expects. Can be 0.
    node_dim : int
        Per-sample output dimension.

    Notes
    -----
    - This class validates shapes in `forward()` then calls `_forward()` which
      subclasses must implement. `_forward()` will always receive the arguments parents and eps.
    """

    def __init__(self, *, input_dim: int, node_dim: int = 1) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError("input_dim must be >= 0.")
        self.input_dim = int(input_dim)
        if node_dim < 0:
            raise ValueError("node_dim must be >= 0.")
        self.node_dim: int = node_dim

    def forward(self, parents: Tensor, eps: Tensor) -> Tensor:
        """
        Parameters
        ----------
        parents : tensor of shape (B, N, D)
        eps : tensor of shape (B, N, E)
        
        where
        D = input_dim, possibly 0
        E = node_dim
        """
        if parents.dim() != 3:
            raise ValueError(f"{self.__class__.__name__}: parents must be 3D (B, N, D). Got {tuple(parents.shape)}.")
        B, N, D = parents.shape
        if D != self.input_dim:
            raise ValueError(f"{self.__class__.__name__}: expected D={self.input_dim}, got D={D}.")

        if eps is not None:
            if eps.dim() != 3:
                raise ValueError(f"{self.__class__.__name__}: noise must be 3D (B, N, E). Got {tuple(eps.shape)}.")
            B, N, E = eps.shape
            if E != self.node_dim:
                raise ValueError(f"{self.__class__.__name__}: expected E={self.node_dim}, got E={E}.")

        y = self._forward(parents, eps)

        expected_out = (B, N, E)
        if tuple(y.shape) != expected_out:
            raise ValueError(
                f"{self.__class__.__name__}: expected output {expected_out}, got {tuple(y.shape)}."
            )
        return y

    @abstractmethod
    def _forward(self, parents: Tensor, eps: Tensor) -> Tensor:
        """Subclasses implement the actual computation. Shapes are validated already."""
        ...