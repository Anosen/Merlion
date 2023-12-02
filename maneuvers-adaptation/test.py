import wandb
import pandas as pd
api = wandb.Api()
run = pd.DataFrame(api.run("/gregoire-marie/MerlionAnomalyDetector/runs/VAE-2023-12-01_00-40-36").history())

run.drop(columns=['_timestamp', '_runtime'], inplace=True)
run.set_index('_step', inplace=True)
print(run.sort_values(by='f2', ascending=False))

print(f'Columns: {run.columns}')

print(f'latent_size: {run['latent_size'].unique()}')
print(f'num_epochs: {run['num_epochs'].unique()}')
print(f'batch_size: {run['batch_size'].unique()}')
print(f'num_eval_samples: {run['num_eval_samples'].unique()}')
print(f'lr: {run['lr'].unique()}')
print(f'sequence_len: {run['sequence_len'].unique()}')

print(run[(run['latent_size']==11) & (run['sequence_len']==1) & (run['num_eval_samples']==5) & (run['lr']==0.0005)].sort_values(by='f2', ascending=False))