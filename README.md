## About

This repository contains code for a prior for a regression prior-fitted network [1]. A predecessor of this prior is the prior developed for Do-PFN [2]. It was refactored and expanded by Arik Reuter (private communication), before being trimmed down and brought into its current form by me. The activation functions used are similar to those used in TabICL [3].

## Features
- an SCM-based prior, with implementation closely following the specification of an SCM by structural equations
- simple code base that should make it relatively easy to adapt things as needed
- easy configuration of basic parameters with configuration file
- compatible out of the box with NanoTabPFN [4]
- fully seeded: on-the-fly data generation is completely determined by one integer seed

## Usage
The following toy example trains a NanoTabPFN Regressor on this prior:
```python
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
```

### References
1. **Müller, S.**, **Hollmann, N.**, **Pineda Arango, S.**, **Grabocka, J.**, **Hutter, F.** *Transformers Can Do Bayesian Inference*. arXiv preprint arXiv:2112.10510, 2021. [https://arxiv.org/abs/2112.10510](https://arxiv.org/abs/2112.10510)  
2. **Robertson, J.**, **Reuter, A.**, **Guo, S.**, **Hollmann, N.**, **Hutter, F.**, **Schölkopf, B.** *Do-PFN: In-Context Learning for Causal Effect Estimation*. arXiv preprint arXiv:2506.06039, 2025. [https://arxiv.org/abs/2506.06039](https://arxiv.org/abs/2506.06039)  
3. **Qu, J.**, **Holzmüller, D.**, **Varoquaux, G.**, **Le Morvan, M.** *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*. arXiv preprint arXiv:2502.05564, 2025. [https://arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)  
4. **Automl**. *TFM-Playground*. GitHub repository, 2024. Available at: [https://github.com/automl/TFM-Playground](https://github.com/automl/TFM-Playground)
