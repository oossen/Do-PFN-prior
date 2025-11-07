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