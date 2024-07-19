import torch
from torch import tensor
import numpy

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


def squared_euclidean_norm(x: torch.Tensor, dim_features: int = None) -> torch.Tensor:
    """
    squared_euclidean_norm
    """
    if dim_features is not None:
        return torch.sum(x * x, dim=dim_features)
    else:
        if x.ndim==0 or x.ndim==1:
            return torch.sum(x * x)
        elif x.ndim==2:
            return torch.sum(x * x, dim=1)
        else:
            raise("enter specify dimension of features in your input tensor.")


def add_to_mean(mean_var, n_mean_samples, new_vars):
    """
    mean_vars: tensor.shape([features])
    n_mean_samples: int
    new_vars: tensor([samples, features])
    """
    new_vars_sum = new_vars.sum(0) if new_vars.ndim==2 else torch.sum(new_vars)
    new_vars_len = new_vars.shape[0] if new_vars.ndim==2 else torch.numel(new_vars)

    return (mean_var * n_mean_samples + new_vars_sum) / (n_mean_samples + new_vars_len)