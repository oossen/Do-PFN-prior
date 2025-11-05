from functools import partial
import torch

from nanotabpfn.utils import get_default_device
from nanotabpfn.train import train
from nanotabpfn.model import NanoTabPFNModel

from utils.bar_distribution import make_bar_distribution
from dataloaders.observational_dataloader import ObservationalDataLoader
from configs.default_config import prior_config


device = get_default_device()

prior = ObservationalDataLoader(num_steps=100, batch_size=2, prior_config=prior_config, seed=42)

model = NanoTabPFNModel(num_attention_heads=8, embedding_size=192, mlp_hidden_size=768, num_layers=6, num_outputs=100)
n_buckets = 100
dist_prior_factory = partial(ObservationalDataLoader, batch_size=10, prior_config=prior_config, seed=42)
dist, buckets = make_bar_distribution(dist_prior_factory, n_buckets=n_buckets, n_samples=1000)

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=3,
    accumulate_gradients=1,
    lr=1e-4,
    device=torch.device(device),
    callbacks=[],
)