from typing import List
from datetime import datetime
from pfns.bar_distribution import get_bucket_limits

from dopfnprior.utils.callbacks import ValidationCallback
from dopfnprior.utils.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback
from tfmplayground.model import NanoTabPFNModel

from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
from dopfnprior.configs.default_config import prior_config


device = get_default_device()
seed = 42
nll = True

prior = ObservationalDataLoader(num_steps=10000,
                                batch_size=4,
                                prior_config=prior_config,
                                seed=seed)

n_buckets = 1000
low, high = -10.0, 10.0
buckets = get_bucket_limits(num_outputs=n_buckets, full_range=(low, high))
model = NanoTabPFNModel(num_attention_heads=8, embedding_size=192, mlp_hidden_size=768, num_layers=6, num_outputs=n_buckets)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
run_name = f"pfn_{'nll' if nll else 'cel'}_{datetime_str}"
output_dir = f"workdir/{run_name}"
tensorboard_dir = f"{output_dir}/tensorboard"
validation_callback = ValidationCallback(tensorboard_dir, prior_config, num_steps=2000)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, validation_callback]
    
trained_model, loss = train(
    model=model,
    prior=prior,
    buckets=buckets.to(device),
    epochs=120,
    lr=1e-4,
    accumulate_gradients=1,
    nll=nll,
    callbacks=callbacks,
    run_name=run_name,
)