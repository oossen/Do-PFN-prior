from datetime import datetime
from pfns.bar_distribution import get_bucket_limits

from dopfnprior.configs.default_config import prior_config
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"visualization/{datetime_str}"

prior = ObservationalDataLoader(50, 1, prior_config, 42)
n_buckets = 2000
low, high = -5.0, 5.0
buckets = get_bucket_limits(num_outputs=n_buckets, full_range=(low, high))
bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0

for i, data in enumerate(prior):
    scm = data['scm']
    sample_shape = (1, 5) # (batch_size, n_rows)
    scm.sample_noise(sample_shape)
    values = scm.propagate()
    log_probs = scm.log_likelihood_batch(values, bucket_mids.unsqueeze(0).expand(1, -1), plot_dir=f"{output_dir}/sample_{i}")
    # write SCM to file
    with open(f"{output_dir}/sample_{i}/scm.txt", "w") as f:
        f.write(str(scm.dag.edges()))
        f.write("\n")
        f.write(str(scm.mechanisms))