import torch
from torch import nn
import time
import schedulefree
import os
from tfmplayground.callbacks import Callback
from tfmplayground.utils import get_default_device
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.interface import init_model_from_state_dict_file
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader



def train(model: NanoTabPFNModel,
          prior: ObservationalDataLoader, 
          buckets: torch.Tensor,
          epochs: int,
          lr: float = 1e-4,
          accumulate_gradients: int = 1,
          nll: bool = False,
          callbacks: list[Callback] = [], 
          run_name: str = 'pfn',
          ckpt_path: str | None = None):
    """
    Trains our model on the given prior using the given criterion.

    Parameters
    ----------
    model : NanoTabPFNModel
        The model to train.
    prior: ObservationalDataLoader
        A dataloader providing training data in the necessary format.
    buckets: torch.Tensor
        The buckets to which the model's outputs will be fit.
    epochs : int
        The number of epochs to train for. One epoch consists of on iteration over `prior`.
    lr : float
        The learning rate.
    accumulate_gradients : int
        The number of batches to accumulate gradients for before performing an optimizer step.
    nll : bool
        Whether to train against just one class label per dataset and test sample (corresponding to the value of `y`).
        In other words, train using NLL instead of cross-entropy loss with class proabilities.
    callbacks : List[Callback]
        A list of callback instances to execute at the end of each epoch (e.g. logging, validation).
    run_name : str
        The name of this training run. Used to create a folder saving the trained model.
    ckpt_path : str | None
        If not None, the path to a checkpoint to load the model, optimizer, and epoch to resume training.
    """
    work_dir = 'workdir/'+run_name
    os.makedirs(work_dir, exist_ok=True)
    device = get_default_device()
    loss_fn = nn.CrossEntropyLoss()
    
    start_epoch = 1
    if ckpt_path is not None:
        model = init_model_from_state_dict_file(ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        start_epoch = state_dict['epoch'] + 1
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0)
    if ckpt_path is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    buckets = buckets.to(device)
    bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()
            model.train()
            optimizer.train()
            total_loss = 0.
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']
                data = (full_data['x'].to(device),
                        full_data['y'][:, :single_eval_pos].to(device))
                if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
                    continue
                
                output = model(data, single_eval_pos=single_eval_pos)
                output = output.view(-1, output.shape[-1])
                
                if nll:
                    y_values = full_data['y'][:, single_eval_pos:].to(device)
                    y_values = y_values.reshape((-1,))
                    # if there are 1001 bucket borders (1000 buckets), clamp to [0, 999]
                    targets = (torch.bucketize(y_values, buckets) - 1).clamp(0, buckets.size(0) - 2)
                else:
                    test_data = {v: full_data['data'][v][:, single_eval_pos:].to(device) for v in full_data['data']}
                    scm = full_data['scm']
                    batch_size = data[0].shape[0]
                    log_probs = scm.log_likelihood_batch(test_data, bucket_mids.unsqueeze(0).expand(batch_size, -1))
                    probs = torch.exp(log_probs)
                    targets = probs.to(device)
                    # renormalize targets from density values to discrete probabilities
                    targets = targets / targets.sum(dim=-1, keepdim=True)
                    targets = targets.view(-1, targets.shape[-1])

                losses = loss_fn(output, targets)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            model.eval()
            optimizer.eval()

            training_state = {
                'seed': prior.seed,
                'epoch': epoch,
                'architecture': {
                    'num_layers': int(model.num_layers),
                    'embedding_size': int(model.embedding_size),
                    'num_attention_heads': int(model.num_attention_heads),
                    'mlp_hidden_size': int(model.mlp_hidden_size),
                    'num_outputs': int(model.num_outputs)
                },
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(training_state, work_dir+'/latest_checkpoint.pth')

            for callback in callbacks:
                callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model, buckets=buckets)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return model, total_loss