## About

This repository contains code for a prior for a regression prior-fitted network [1]. A predecessor of this prior is the prior developed for Do-PFN [2]. It was refactored and expanded by Arik Reuter (private communication), before being trimmed down and brought into its current form by me. The activation functions used are similar to those used in TabICL [3].

## Features
- an SCM-based prior, with implementation closely following the specification of an SCM by structural equations
- simple code base that should make it relatively easy to adapt things as needed
- easy configuration of basic parameters with configuration file
- compatible out of the box with the NanoTabPFN training loop from TFM-Playground [4]
- fully seeded: on-the-fly data generation is completely determined by one integer seed

## Usage
The following toy example trains a NanoTabPFN Regressor on this prior:
```python
import torch
from sklearn.metrics import r2_score

from tfmplayground.utils import get_default_device
from tfmplayground.train import train
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.interface import NanoTabPFNRegressor

from dopfnprior.utils.bar_distribution import make_bar_distribution
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
from dopfnprior.configs.default_config import prior_config


device = get_default_device()

prior = ObservationalDataLoader(num_steps=1000, batch_size=2, prior_config=prior_config, seed=42)

n_buckets = 1000
model = NanoTabPFNModel(num_attention_heads=8, embedding_size=192, mlp_hidden_size=768, num_layers=6, num_outputs=n_buckets)
dist_prior = ObservationalDataLoader(1000, batch_size=10, prior_config=prior_config, seed=43)
dist, buckets = make_bar_distribution(dist_prior, n_buckets=n_buckets)

class ValidationLoggerCallback(ConsoleLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        # this prior contains the exact same data every time it is created
        validation_prior = ObservationalDataLoader(num_steps=100, batch_size=1, prior_config=prior_config, seed=44)
        dist = kwargs['dist']
        regressor = NanoTabPFNRegressor(model, dist, get_default_device())
        scores = []
        for data in validation_prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            
            regressor.fit(X_train, y_train)
            pred = regressor.predict(X_test)
            scores.append(r2_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        print(f'Average R² score on validation data {avg_score:.3f}')

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=100,
    accumulate_gradients=1,
    lr=1e-4,
    device=torch.device(device),
    callbacks=[ConsoleLoggerCallback(), ValidationLoggerCallback()],
)
```

### References
1. **Müller, S.**, **Hollmann, N.**, **Pineda Arango, S.**, **Grabocka, J.**, **Hutter, F.** *Transformers Can Do Bayesian Inference*. arXiv preprint arXiv:2112.10510, 2021. [https://arxiv.org/abs/2112.10510](https://arxiv.org/abs/2112.10510)  
2. **Robertson, J.**, **Reuter, A.**, **Guo, S.**, **Hollmann, N.**, **Hutter, F.**, **Schölkopf, B.** *Do-PFN: In-Context Learning for Causal Effect Estimation*. arXiv preprint arXiv:2506.06039, 2025. [https://arxiv.org/abs/2506.06039](https://arxiv.org/abs/2506.06039)  
3. **Qu, J.**, **Holzmüller, D.**, **Varoquaux, G.**, **Le Morvan, M.** *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*. arXiv preprint arXiv:2502.05564, 2025. [https://arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)  
4. **Automl**. *TFM-Playground*. GitHub repository, 2024. Available at: [https://github.com/automl/TFM-Playground](https://github.com/automl/TFM-Playground)
