### A try to train on multiple TS at a time
### FIXME


from functools import reduce
import operator
import sys
import logging
import unittest
import torch
import random
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
from ts_datasets.anomaly import *
import json
import itertools
from tqdm import tqdm
import pprint
from torchsummary import summary
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.factory import ModelFactory

from merlion.utils import TimeSeries
from merlion.models.anomaly.autoencoder import AutoEncoder, AutoEncoderConfig
from merlion.models.anomaly.vae import VAE, VAEConfig
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig


rootdir = dirname(abspath(__file__))
logger = logging.getLogger(__name__)


class ManeuverDetection():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_types = ['AutoEncoder', 'VAE', 'IsolationForest']
        self.model = None
        self.dataset = CustomAnomalyDataset(
            rootdir=join(rootdir, 'data', 'residues'),   # where the data is stored
            test_frac=0.70          # use 100*test_frac % of each time series for testing.
                                    # overridden if the column `trainval` is in the actual CSV.
            #time_unit="s",          # the timestamp column (automatically detected) is in units of seconds
            #assume_no_anomaly=True  # if a CSV doesn't have the "anomaly" column, assume it has no anomalies
        )

    def get_ts(self, i: int) -> (pd.DataFrame, pd.DataFrame):
        self.df, self.metadata = self.dataset[0]
        return self.df, self.metadata
    
    def get_train_test_splits(self, df: pd.DataFrame, metadata: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
        self.train_df = df[metadata.trainval]
        self.test_df = df[~metadata.trainval]
        self.test_labels = metadata.anomaly[~metadata.trainval]
        return self.train_df, self.test_df, self.test_labels
    
    def get_concat_ts(self):
        all_train_dfs = []
        all_test_dfs = []
        all_test_labels_dfs = []

        for i in range(len(self.dataset)):
            df, metadata = self.get_ts(i)
            train_df, test_df, test_labels = self.get_train_test_splits(df, metadata)

            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            all_test_labels_dfs.append(test_labels)

        # Concatenate all DataFrames
        concatenated_train_df = pd.concat(all_train_dfs, ignore_index=True)
        concatenated_test_df = pd.concat(all_test_dfs, ignore_index=True)
        concatenated_test_labels = pd.concat(all_test_labels_dfs)

        return concatenated_train_df, concatenated_test_df, concatenated_test_labels
    
    def get_list_ts(self):
        all_train_dfs = []
        all_test_dfs = []
        all_test_labels_dfs = []

        for i in range(len(self.dataset)):
            df, metadata = self.get_ts(i)
            train_df, test_df, test_labels = self.get_train_test_splits(df, metadata)

            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            all_test_labels_dfs.append(test_labels)

        return all_train_dfs, all_test_dfs, all_test_labels_dfs

    def test_score(self):
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        test_ts = TimeSeries.from_pd(self.test_df)

        score_ts = self.model.get_anomaly_score(test_ts)
        scores = score_ts.to_pd().values.flatten()
        min_score, max_score, sum_score = min(scores), max(scores), sum(scores)

        logger.info(f"scores look like: {scores[:10]}")
        logger.info(f"min score = {min_score}")
        logger.info(f"max score = {max_score}")
        logger.info(f"sum score = {sum_score}")

    def save_model(self):
        print("-" * 80)
        logger.info("save_model\n" + "-" * 80 + "\n")
        self.model.save(dirname=join(rootdir, "tmp", "vae"))
        logger.debug(f'Saved {self.model.__module__} config:\n{self.model.config.to_dict()}')
        
    def load_model(self):
        loaded_model = type(self.model).load(dirname=join(rootdir, "tmp", "vae"))

        test_ts = TimeSeries.from_pd(self.test_df)
        scores = self.model.get_anomaly_score(test_ts)
        loaded_model_scores = loaded_model.get_anomaly_score(test_ts)
        self.assertSequenceEqual(list(scores), list(loaded_model_scores))

        alarms = self.model.get_anomaly_label(test_ts)
        loaded_model_alarms = loaded_model.get_anomaly_label(test_ts)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))

def main():
    print(logger)
    
    md = ManeuverDetection()

    # Load hyperparams_dict from the JSON file
    with open(join(rootdir, 'conf', 'maneuver_detection.json'), 'r') as json_file:
        hyperparams_dict = json.load(json_file)

    for model_type in tqdm(md.model_types):
        hyperparams = hyperparams_dict[model_type]
        hyperparams_count = {key: len(value) if isinstance(value, (list, np.ndarray)) else 1 for key, value in hyperparams.items()}
        total_hyperparams_count = reduce(operator.mul, hyperparams_count.values(), 1)

        logger.info(f'All hyperparams for {model_type}:\n{pprint.pformat(hyperparams)}\n\nHyperparams count:\n{pprint.pformat(hyperparams_count)}\n\nTotal iterations: {total_hyperparams_count}\n\nStarting optimization...')
        # Convert some lists to tuple
        if model_type == 'VAE':
            hyperparams['encoder_hidden_sizes'] = [tuple(param) for param in hyperparams['encoder_hidden_sizes']]
        if model_type == 'AutoEncoder':
            hyperparams['layer_sizes'] = [tuple(param) for param in hyperparams['layer_sizes']]

        # Generate all combinations of hyperparameters
        hyperparam_combinations = list(itertools.product(*hyperparams.values()))
        
        # Iterate over each combination
        for i, combination in enumerate(tqdm(hyperparam_combinations), start=1):
            ## Fetch the model current configuration from the hyperparameters list
            model_kwargs = dict(zip(hyperparams.keys(), combination))
            if model_type == 'VAE': # Add the 'decoder_hidden_sizes' hyperparameter, symetric of 'encoder_hidden_sizes'
                model_kwargs['decoder_hidden_sizes'] = model_kwargs['encoder_hidden_sizes'][::-1]
            
            ## Update the model with the new hyperparameters
            model = ModelFactory.create(name=model_type, **model_kwargs)
                        
            logger.warning(f"{i}/{len(hyperparam_combinations)} - Model {model_type}:\n{pprint.pformat(model_kwargs)}")

            ## Get the long TS, concatenation of all TS
            all_train_df, all_test_df, all_test_labels_df = md.get_list_ts()
            
            #summary(model, concatenated_train_df.shape)

            train_ts = [TimeSeries.from_pd(df) for df in all_train_df]
            test_ts = [TimeSeries.from_pd(df) for df in all_test_df]
            test_labels_ts = [TimeSeries.from_pd(df) for df in all_test_labels_df]
            
            # Train model
            model.train_multipleTS(train_data=train_ts, max_length=500)
            
            # Save model
            save_dir = join(rootdir, 'results', 'models', model_type, str(i))
            model.save(save_dir)
            
            # Evaluate model
            labels = model.get_anomaly_label(test_ts)
            precision = TSADMetric.PointAdjustedPrecision.value(ground_truth=test_labels_ts, predict=labels)
            recall = TSADMetric.PointAdjustedRecall.value(ground_truth=test_labels_ts, predict=labels)
            f1 = TSADMetric.PointAdjustedF1.value(ground_truth=test_labels_ts, predict=labels)
            f2 = TSADMetric.F2.value(ground_truth=test_labels_ts, predict=labels)
            mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels_ts, predict=labels)
            
            # Save metrics
            metrics = {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "F2": f2,
            }
            metrics_json = join(save_dir, 'metrics.json')
            with open(metrics_json, "w") as f:
                json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s %(asctime)s (%(module)s:%(lineno)d):\n%(message)s", stream=sys.stdout, level=logging.INFO
    )
    main()