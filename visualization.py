from datetime import datetime
import torch
from matplotlib import pyplot as plt
import os

from pfns.bar_distribution import get_bucket_limits, FullSupportBarDistribution
from tfmplayground.interface import init_model_from_state_dict_file
from dopfnprior.configs.default_config import prior_config
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
    
now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"visualization/{datetime_str}"
os.makedirs(output_dir, exist_ok=True)
    
seed = 100
generator = torch.Generator()
generator.manual_seed(seed)
    
prior = ObservationalDataLoader(50, 1, prior_config, seed=seed)

n_buckets = 2000
low, high = -5.0, 5.0
buckets = get_bucket_limits(num_outputs=n_buckets, full_range=(low, high)).to('cpu')
bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0
dist = FullSupportBarDistribution(buckets)
    
model_names = [{"name": "pfn_cel_03_14_01_36", "color": "red", "label": "p(y|x, D) (CEL)"},
          {"name": "pfn_nll_03_14_01_37", "color": "blue", "label": "p(y|x, D) (NLL)"}]

for i, data in enumerate(prior):
    X = data['x'].cpu()  # shape (1, N, F)
    y = data['y'].cpu()  # shape (1, N,, 1)
    single_eval_pos = data['single_eval_pos']
    y_train = y[:single_eval_pos]
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
    y_norm = (y_train - y_mean) / y_std
    scm = data['scm']
    fig, ax = plt.subplots(figsize=(8, 5))
    xlim_set = False
    for model_name in model_names:
        model_path = f"workdir/{model_name['name']}/latest_checkpoint.pth"
        model = init_model_from_state_dict_file(model_path)
        model.eval()
        model.to('cpu')
        with torch.no_grad():
            logits = model((X, y_norm), single_eval_pos=single_eval_pos)
            logits = logits[0][0] # only look at one sample
            logits = logits.view(1, 1, -1).expand(len(bucket_mids), 1, -1)
            neg_log_probs = dist.forward(logits, bucket_mids).squeeze(0).squeeze(-1)
            probs = torch.exp(-neg_log_probs)
            eps = 0.01 * probs.max()
            mask = probs > eps
            indices = torch.where(mask)[0]
            buffer = 1
            start_idx = max(0, indices[0] - buffer)
            end_idx = min(len(bucket_mids) - 1, indices[-1] + buffer)
            a = bucket_mids[start_idx].item()
            b = bucket_mids[end_idx].item()
            ax.plot(bucket_mids, probs.cpu(), color=model_name['color'], label=model_name['label'])
            curr_min, curr_max = ax.get_xlim()
            if not xlim_set:
                ax.set_xlim(a, b)
                xlim_set = True
            else:
                ax.set_xlim(min(curr_min, a), max(curr_max, b))
    plt.savefig(f"{output_dir}/pfn_predictions_{i}.png")
    plt.close(fig)