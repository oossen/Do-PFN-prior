from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
from tfmplayground.callbacks import TensorboardLoggerCallback
from tfmplayground.utils import get_default_device

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error


class ValidationCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, prior_config, num_steps=1000, seed=100):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_config = prior_config
        self.num_steps = num_steps
        self.seed = seed
    
    @torch.no_grad()
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = ObservationalDataLoader(self.num_steps, 1, self.prior_config, self.seed)
        device = get_default_device()
        buckets = kwargs['buckets'].to(device)
        bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0
        
        r2_scores = []
        mse_scores = []
        nll_scores = []
        cel_scores = []
        
        for data in test_prior:
            X = data['x'].to(device)
            y = data['y'].to(device)
            single_eval_pos = data['single_eval_pos']
            y_train = y[:, :single_eval_pos]
            y_mean = y_train.mean(dim=1, keepdim=True)
            y_std = y_train.std(dim=1, keepdim=True) + 1e-8
            y_norm = (y_train - y_mean) / y_std
            y_target = y[:, single_eval_pos:]
            y_target = (y_target - y_mean) / y_std
            y_target = y_target.reshape((-1,))
            y_target_buckets = (torch.bucketize(y_target, buckets) - 1).clamp(0, buckets.size(0) - 2)
            
            test_data = {v: data['data'][v][:, single_eval_pos:] for v in data['data']}
            scm = data['scm']
            buckets_rescaled = buckets.unsqueeze(0) * y_std.squeeze(-1) + y_mean.squeeze(-1)
            log_probs = scm.log_likelihood_batch(test_data, buckets_rescaled)
            probs = torch.exp(log_probs).to(device)
            y_target_dist = probs / probs.sum(dim=-1, keepdim=True)
            y_target_dist = y_target_dist.view(-1, y_target_dist.shape[-1])
            
            logits = model((X, y_norm), single_eval_pos=single_eval_pos)
            logits = logits.view(-1, logits.shape[-1])
            probs = torch.softmax(logits, dim=-1)
            y_pred = probs @ bucket_mids
            
            ce_loss = torch.nn.CrossEntropyLoss()
            nll = ce_loss(logits, y_target_buckets).item()
            cel = ce_loss(logits, y_target_dist).item()
            
            r2 = r2_score(y_target.cpu().numpy(), y_pred.cpu().numpy())
            mse = mean_squared_error(y_target.cpu().numpy(), y_pred.cpu().numpy())
            
            r2_scores.append(r2)
            mse_scores.append(mse)
            nll_scores.append(nll)
            cel_scores.append(cel)
            
        avg_r2 = sum(r2_scores) / len(r2_scores)
        avg_mse = sum(mse_scores) / len(mse_scores)
        avg_nll = sum(nll_scores) / len(nll_scores)
        avg_cel = sum(cel_scores) / len(cel_scores)
        self.writer.add_scalar('Validation R²', avg_r2, epoch)
        self.writer.add_scalar('Validation MSE', avg_mse, epoch)
        self.writer.add_scalar('Validation NLL', avg_nll, epoch)
        self.writer.add_scalar('Validation CEL', avg_cel, epoch)