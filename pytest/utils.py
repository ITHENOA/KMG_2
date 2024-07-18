import torch
from torch import tensor

# k: int = 1
similarity_thresh = torch.exp(tensor([-3]))


def get_device() -> str:
    """
    Get the device to be used for computation.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device


def squared_euclidean_norm(x: torch.Tensor) -> torch.Tensor:
    """
    squared_euclidean_norm
    """
    return torch.sum(x * x, dim=1) if x.ndim == 2 else torch.sum(x * x)
