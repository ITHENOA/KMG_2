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


def add_to_mean(mean_var, n_mean_samples, new_vars):
    """
    mean_vars: tensor.shape([features])
    n_mean_samples: int
    new_vars: tensor([samples, features])
    """
    # if new_vars.ndim == 2:
    #     new_vars_sum = new_vars.sum(0)
    #     new_vars_len = new_vars.shape[0]
    # else:
    #     new_vars_sum = torch.sum(new_var)
    #     new_vars_len = len(new_vars)
    new_vars_sum = new_vars.sum(0) if new_vars.ndim==2 else torch.sum(new_vars)
    new_vars_len = new_vars.shape[0] if new_vars.ndim==2 else torch.numel(new_vars)

    return (mean_var * n_mean_samples + new_vars_sum) / (n_mean_samples + new_vars_len)