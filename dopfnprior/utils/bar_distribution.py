from pfns.bar_distribution import FullSupportBarDistribution
from pfns.bar_distribution import get_bucket_limits
from tfmplayground.utils import get_default_device

import torch
from torch.utils.data import DataLoader


def make_bar_distribution(prior: DataLoader, n_buckets: int = 100):
    """
    Construct a full support bar/Riemann distribution.
    Sample `n_samples` many data tables from `prior` and
    choose `n_buckets` many buckets to each contain the same number of targets.
    
    The argument `prior_factory` takes one int argument (number of samples) and returns a dataloader.
    """
    sampled_ys = []
    for data in prior:
        y = data['y']
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True) + 1e-8
        y_norm = (y - y_mean) / y_std
        for col in y_norm: # only one iteration if batch_size==1
            sampled_ys.append(col.unsqueeze(-1))
    ys_tensor = torch.concat(sampled_ys)
    
    device = get_default_device()
    buckets = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return FullSupportBarDistribution(buckets), buckets