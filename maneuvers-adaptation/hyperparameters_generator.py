from functools import reduce
import operator
import numpy as np
import json
from os.path import abspath, dirname, join

rootdir = dirname(abspath(__file__))

## AE
ae_hyperparams = { # A dict to store the hyperparameter search space
    'hidden_size': list(range(10,13,1)),
    'layer_sizes': [(15, 10, 5)],
    'sequence_len': list(range(1,4,1)),
    'lr': [0.5e-3, 1.0e-3, 5e-3],
    'batch_size': [512],
    'num_epochs': [50]#[25, 50, 100]
}

## VAE
vae_hyperparams = { # A dict to store the hyperparameter search space
    'encoder_hidden_sizes': [(15, 10, 5)],
    #'decoder_hidden_sizes': *invert of encoder_hidden_sizes*,
    'latent_size': list(range(10,13,1)),
    'sequence_len': list(range(1,4,1)),
    'kld_weight': [1.0],
    'dropout_rate': [0.0],
    'num_eval_samples': list(range(5,7,1)),
    'lr': [0.5e-3, 1.0e-3],
    'batch_size': [1024],
    'num_epochs': [20]#[5, 15, 25],
}

## LSTMED
lstmed_hyperparams = { # A dict to store the hyperparameter search space
    'hidden_size': list(range(1,20,5)),
    'sequence_len': list(range(10,30,5)),
    'n_layers': [(1, 1)],
    'dropout': [(0, 0)],
    'lr': [0.5e-3, 1.0e-3, 5e-3, 1.0e-2],
    'batch_size': [256, 512],
    'num_epochs': [30]
}

## Isolation Forest
isolfor_hyperparams = { # A dict to store the hyperparameter search space
    'max_n_samples': [0.1, 0.3, 0.5, 0.7, 1],
    'n_estimators': list(range(20,300,50)),
}

## A dict for hyperparameters of all models
hyperparams_dict = {
    'AutoEncoder': ae_hyperparams, 
    'VAE': vae_hyperparams,
    'LSTMED': lstmed_hyperparams,
    'IsolationForest': isolfor_hyperparams
    }

for key, value in hyperparams_dict.items():
    result = {key: len(value) if isinstance(value, (list, np.ndarray)) else 1 for key, value in value.items()}
    total_product = reduce(operator.mul, result.values(), 1)
    print(f'Hyperparams for {key}:\n{value}')
    print(f'Hyperparams count:\n{result}')
    print(f'Total iterations: {total_product}\n')
    
# Save hyperparams_dict as a JSON file
save_file = join(rootdir, 'conf', 'hyperparam_optimization.json')
with open(save_file, 'w') as json_file:
    json.dump(hyperparams_dict, json_file, indent=2)

print(f"Hyperparameters saved to {save_file}")