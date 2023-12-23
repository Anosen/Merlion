#!pip install pandas matplotlib

from collections import OrderedDict
import copy
import os
import torch
import numpy as np
from scipy.stats import norm  
from scipy.stats import truncnorm


import wandb
import logging
from os.path import abspath, dirname, join
import itertools
from tqdm import tqdm
import pprint
from pathlib import Path
from datetime import timedelta, datetime
import json
import itertools
import csv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from merlion.models.anomaly.autoencoder import AutoEncoder
from merlion.models.anomaly.vae import VAE
from merlion.models.anomaly.isolation_forest import IsolationForest
from merlion.models.anomaly.lstm_ed import LSTMED
from merlion.evaluate.anomaly import TSADMetric
from ts_datasets.anomaly import CustomAnomalyDataset
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.utils import TimeSeries
from merlion.utils.misc import ProgressBar
from merlion.utils.misc import call_with_accepted_kwargs
from merlion.plot import plot_anoms
from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType


rootdir = dirname(abspath(__file__))
run_id = '2023-12-13_15-25-56'
# run_id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

class DummyLogger:
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            pass
        return dummy_method

def convert_tensors_to_numpy(d):
    if isinstance(d, dict):
        return {key: convert_tensors_to_numpy(value) for key, value in d.items()}
    elif isinstance(d, torch.Tensor):
        # If it's a tensor, use the numpy() method to convert to NumPy array
        return d.detach().cpu().numpy().tolist()
    else:
        return d

def get_test(dataset, entry, normalize=False):
    time_series, metadata = dataset[entry]

    test_data = TimeSeries.from_pd(time_series)
    test_labels = TimeSeries.from_pd(metadata.anomaly)
    return test_data, test_labels

def split_csv(root_csv_dir, split_size):
    test_csv_dir = join(root_csv_dir, 'test_residues')
    train_val_csv_dir = join(root_csv_dir, 'train_val_residues')
    
    csv_files = [file for file in os.listdir(root_csv_dir) if file.endswith(".csv")]

    for csv_file in csv_files:
        print(csv_file)
        test_file = os.path.join(test_csv_dir, f'test_{csv_file}')
        train_val_file = os.path.join(train_val_csv_dir, f'train_val_{csv_file}')
        
        with open(join(root_csv_dir, csv_file), 'r') as infile, open(test_file, 'a', newline='') as testFile, open(train_val_file, 'a', newline='') as train_valFile:
            reader = csv.reader(infile)
            test_writer = csv.writer(testFile)
            train_val_writer = csv.writer(train_valFile)
            
            total_lines = sum(1 for row in reader)
            train_val_lines = int(total_lines * split_size)
            
            infile.seek(0)  # Rewind the input file

            # Erase the content of the output files
            testFile.truncate(0)
            train_valFile.truncate(0)
            
            for i, row in enumerate(reader):
                if i==0:
                    test_writer.writerow(row)
                    train_val_writer.writerow(row)
                elif 0< i <= train_val_lines:
                    train_val_writer.writerow(row)
                else:
                    test_writer.writerow(row)
                    
        infile.close()
        testFile.close()
        train_valFile.close()
            
    return train_val_csv_dir, test_csv_dir

def save_checkpoint(model, kargs, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'kargs': kargs
    }
    torch.save(checkpoint, save_path)

class ManeuverDetection():

    def __init__(self, dataset, log_wandb=False, evaluate=True, log=True, save_log=False):

        if log:
            ## Create a logger
            self.logger = logging.getLogger('ManeuverDetectionLogger')
            self.logger.setLevel(logging.DEBUG)
            # Create a console handler
            self.console_handler = logging.StreamHandler() #sys.stdout
            # Create a formatter
            self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - (%(module)s:%(lineno)d): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            # Add the formatter to the console handler
            self.console_handler.setFormatter(self.formatter)
            # Add the console handler to the logger
            self.logger.addHandler(self.console_handler)
            if save_log:
                # Create a file handler and set the level to DEBUG
                self.file_handler = logging.FileHandler(join(rootdir, 'results', 'logs', f'{run_id}.log'))
                self.file_handler.setLevel(logging.DEBUG)
                # Add the formatter to the file handler
                self.file_handler.setFormatter(self.formatter)
                # Add the file handler to the logger
                self.logger.addHandler(self.file_handler)
                
                # Set the console_handler level to INFO
                self.console_handler.setLevel(logging.INFO)
                # Add the console handler to the logger
                self.logger.addHandler(self.console_handler)
            else:
                # Set the console_handler level to DEBUG
                self.console_handler.setLevel(logging.DEBUG)
                # Add the console handler to the logger
                self.logger.addHandler(self.console_handler)
        else:
            self.logger = DummyLogger()

        self.logger.info(f'Initializing ManeuverDetection: log_wandb: {log_wandb} - evaluate: {evaluate}')
        self.logger.debug(f'DEBUG')
        self.dataset_dir = join(rootdir, 'data', 'residues')
        self.model_types = {
            'AutoEncoder': type(AutoEncoder(config=AutoEncoder.config_class())),
            'VAE': type(VAE(config=VAE.config_class())),
            'LSTMED': type(LSTMED(config=LSTMED.config_class())),
            'IsolationForest': type(IsolationForest(config=IsolationForest.config_class()))
            }
        
        self.evaluate=evaluate
        self.log_wandb=log_wandb
               
        if dataset is None:
            self.logger.info(f'Fetching dataset: {self.dataset_dir}')
            
            # Set test and validation size, the rest is used for train
            self.test_size = 0.1
            self.validation_size = 0.7
            
            self.train_val_dir, self.test_dataset_dir = split_csv(self.dataset_dir, self.test_size)
            self.test_dataset = CustomAnomalyDataset(
                rootdir=self.test_dataset_dir
            )
            
            self.test_split_dataset = {
                i: {
                    'test_data': TimeSeries.from_pd(time_series),
                    'test_labels': TimeSeries.from_pd(metadata.anomaly)
                }
                for i, (time_series, metadata) in enumerate(self.test_dataset)
            }
 
            self.dataset = CustomAnomalyDataset(
                # rootdir=self.train_val_dir,                       # where the data is stored
                rootdir=self.dataset_dir,                       # where the data is stored
                test_frac=self.validation_size#-self.test_size   # use 100*test_frac % of each time series for validation.
                                                                # overridden if the column `trainval` is in the actual CSV.
                #time_unit="s",                                 # the timestamp column (automatically detected) is in units of seconds
                #assume_no_anomaly=True                         # if a CSV doesn't have the "anomaly" column, assume it has no anomalies
            )
        self.amount_of_entry = len(self.dataset)
        self.test_amount_of_entry = len(self.test_dataset)
        
        # Split the dataset
        self.train_val_split_dataset = self.crop_dataset({
            i: {
                'train_data': TimeSeries.from_pd(time_series[metadata.trainval]),
                'train_labels': TimeSeries.from_pd(metadata.anomaly[metadata.trainval]),
                'val_data': TimeSeries.from_pd(time_series[~metadata.trainval]),
                'val_labels': TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])
            } for i, (time_series, metadata) in enumerate(self.dataset)
        })

        # Load thresholds from the JSON file
        with open(join(rootdir, 'conf', 'alm_thresholds.json'), 'r') as json_file:
            self.thresholds = json.load(json_file)

    def crop_dataset(self, split_dataset):
        for key in split_dataset[0].keys():
            series_lengths = [len(split_dataset[i][key]) for i in range(len(split_dataset))]
            min_series_length = min(series_lengths)
            max_series_length = max(series_lengths)
            self.logger.debug(f'{key} lengths: {series_lengths} | Smallest: {min_series_length}: {series_lengths.index(min_series_length)} | Longest: {max_series_length}: {series_lengths.index(max_series_length)}')
            self.logger.info(f'Cropping {key} to a length of {min_series_length}')
            for i in range(len(split_dataset)):
                split_dataset[i][key] = TimeSeries.from_pd(TimeSeries.to_pd(split_dataset[i][key]).head(min_series_length))
        return split_dataset

    def load_checkpoint(self, model, optimizer, load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # # Convert tensors to NumPy arrays for JSON serialization
        # checkpoint_copy = {}
        # for key, value in checkpoint.items():
        #     if isinstance(value, torch.Tensor):
        #         checkpoint_copy[key] = convert_tensors_to_numpy(value)
        #     elif isinstance(value, OrderedDict):
        #         checkpoint_copy[key] = {key1: convert_tensors_to_numpy(value1) for key1, value1 in value.items()}
        #     elif key == 'optimizer_state_dict':
        #         checkpoint_copy[key] = convert_tensors_to_numpy(value)
        #     else:
        #         checkpoint_copy[key] = value
        # # Save the modified checkpoint dictionary to a JSON file
        # json_path = join(rootdir, "checkpoint.json")
        # with open(json_path, 'w') as json_file:
        #     json.dump(checkpoint_copy, json_file)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        kargs = checkpoint['kargs']
        return model, optimizer, kargs, epoch, loss

    def train(self, model, train_data, train_labels=None):
        train_start = datetime.now()
        ## Train
        train_data = model.train_pre_process(train_data)
        train_data = train_data.to_pd() if model._pandas_train else train_data
        train_result = call_with_accepted_kwargs(  # For ensembles
            model._train, train_data=train_data, train_config=None, anomaly_labels=None
        )
        # train_scores = model.train(train_data, anomaly_labels=train_labels)
        train_time = datetime.now()-train_start
        return train_time, train_result

    def eval(self, model, data, labels, score_type=ScoreType.RevisedPointAdjusted):
        model.model.eval()
        ## Evaluate
        predict = model.get_anomaly_label(data)
        
        ## Accumulate scores
        scores = accumulate_tsad_score(predict=predict, ground_truth=labels, max_early_sec=None, max_delay_sec=None)
        
        ## Compute scores
        precision = scores.precision(score_type=score_type)
        recall = scores.recall(score_type=score_type)
        f1 = scores.f1(score_type=score_type)
        f2 = scores.f_beta(score_type=score_type, beta=2.0)
        mttd = scores.mean_time_to_detect()
        mdad = 0.0#scores.mean_detected_anomaly_duration()
        mad = 0.0#scores.mean_anomaly_duration()
        #self.logger.debug(f'{score_type.name}: Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | MTTD: {mttd} | MDAD: {mdad} | MAD: {mad}')
        return scores, precision, recall, f1, f2, float(mttd.total_seconds())

    def train_eval(self, model, train_data, val_data, val_labels, test_data, test_labels, model_type=None, model_kwargs=None, noradId=None, plot=False):
        # Train the model and infere the train data
        train_time, _ = self.train(model, train_data, train_labels=None)
        
        best_threshold = 0.0
        best_pr = 0.0
        best_f2 = 0.0
        best_roc = 0.0
        
        val_precision_list = []
        val_recall_list = []
        
        tpr_list = []
        fpr_list = []
        
        num_points = 10 # Number of threshold to generate
        thresholds_list = np.linspace(0, 6, num_points)

        # # Set the parameters for the Gaussian distribution
        # mean = 3  # Center of the distribution
        # std_dev = 3  # Standard deviation (controls the spread)

        # min=0 # Start of the distribution

        # # Define the parameters of the truncated normal distribution
        # a = (min - mean) / std_dev  # Lower bound (0 in this case) centered on 3, with a standard deviation of 1
        # b = (np.inf - mean) / std_dev  # Upper bound (positive infinity) centered on 3, with a standard deviation of 1
        # # Generate random samples from the truncated normal distribution
        # thresholds_list = truncnorm.rvs(a, b, loc=3, scale=1, size=num_points)
        # thresholds_list.sort()
        
        for t in thresholds_list:
            model.threshold.alm_threshold = t

            # Get val precision and recall
            val_scores, val_precision, val_recall, _, val_f2, _ = self.eval(model, data=val_data, labels=val_labels)
            
            # TPR = TP / (TP + FN)
            tp_rpa = val_scores.num_tp_anom
            fn_rpa = val_scores.num_fn_anom
            tpr = tp_rpa / (tp_rpa + fn_rpa)
            tpr_list.append(tpr)
            
            # FPR = FP / (FP + TN)
            fp = val_scores.num_fp
            tn = val_scores.num_tn
            fpr = fp / (fp + tn)
            fpr_list.append(fpr)         
            
            # Compute the area under the curve using the trapezoidal rule for the PR curve
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            
            val_roc = (tpr+1-fpr)/2
            if best_roc<val_roc:
                best_roc = val_roc
                best_threshold_roc = t
            
            val_pr = (val_precision+val_recall)/2
            if best_pr<val_pr:
                best_pr = val_pr
                best_threshold = t
                
            if best_f2<val_f2:
                best_f2 = val_f2
                best_threshold_f2 = t

        # Evaluate the model with the best threshold
        # model.threshold.alm_threshold = best_threshold
        # _, test_precision, test_recall, test_f1, test_f2, test_mttd = self.eval(model, test_data, test_labels)
        
        model.threshold.alm_threshold = best_threshold_f2
        _, precision_best_f2, recall_best_f2, f1_best_f2, f2_best_f2, mttd_best_f2 = self.eval(model, test_data, test_labels)
        test_precision, test_recall, test_f1, test_f2, test_mttd = precision_best_f2, recall_best_f2, f1_best_f2, f2_best_f2, mttd_best_f2
        
        # model.threshold.alm_threshold = best_threshold_roc
        # _, precision_best_roc, recall_best_roc, f1_best_roc, f2_best_roc, mttd_best_roc = self.eval(model, test_data, test_labels)

        fpr_list, tpr_list = (list(t) for t in zip(*sorted(zip(fpr_list, tpr_list)))) # Order the lists in order of the abscissa
        val_aucroc = np.trapz(tpr_list, fpr_list) # Compute the AUCROC using trapezoidal method
        
        val_recall_list, val_precision_list = (list(t) for t in zip(*sorted(zip(val_recall_list, val_precision_list)))) # Order the lists in order of the abscissa
        val_aucpr = np.trapz(val_precision_list, val_recall_list) # Compute the AUCPR using trapezoidal method

        if plot:
            model_str_args = '-'.join(['{1}'.format(k, v) for k,v in model_kwargs.items() if k!='alm_threshold'])
            scatter_roc = plt.scatter(fpr_list, tpr_list, c=thresholds_list, cmap='viridis') # Plot ROC Curve
            plt.colorbar(scatter_roc).set_label('Thresholds')
            plt.title(f'ROC of {noradId} | AUC: {val_aucroc}\n{model_str_args}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # Define the directory path
            roc_model_directory = os.path.join(rootdir, 'results', 'roc', model_type, model_str_args)
            os.makedirs(roc_model_directory, exist_ok=True)  # Create the directory if it doesn't exist
            plt.savefig(join(roc_model_directory, f'roc_{noradId}_{model_str_args}.png'))
            plt.clf()
            
            scatter_pr = plt.scatter(val_recall_list, val_precision_list, c=thresholds_list) # Plot PR Curve
            plt.colorbar(scatter_pr).set_label('Thresholds')
            plt.title(f'PR of {noradId} | AUC: {val_aucpr}\n{model_str_args}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # Define the directory path
            pr_model_directory = os.path.join(rootdir, 'results', 'pr', model_type, model_str_args)
            os.makedirs(pr_model_directory, exist_ok=True)  # Create the directory if it doesn't exist
            plt.savefig(join(pr_model_directory, f'pr_{noradId}_{model_str_args}.png'))
            plt.clf()

        ## Print each univariate anomaly score and ground truth labels
        # print(f'Available univariates: {train_data.names}')
        # for i in range(len(train_data.names)):
        #     plt.close()
        #     fig, ax = model.plot_anomaly(time_series=test_data, index=i)
        #     plot_anoms(ax=ax, anomaly_labels=test_labels)
        #     plt.show() 

        return test_precision, test_recall, test_f1, test_f2, test_mttd, float(train_time.total_seconds()), best_threshold, val_aucroc

    def threshold_adapt(self, model_kwargs, train_data):
        train_start_date = datetime.fromtimestamp(train_data.t0)
        tmp_thresholds = [value['alm_threshold'] for key, value in self.thresholds.items()]
        avg_thresholds = sum(tmp_thresholds) / len(tmp_thresholds)
        for key, value in self.thresholds.items():
            start_date = datetime.strptime(value['start_date'], "%Y-%m-%d")
            error_margin = timedelta(1)
            if (train_start_date > start_date - error_margin) and (train_start_date < start_date + error_margin):
                model_kwargs['alm_threshold']=value['alm_threshold']
                noradid=key
                break
            else:
                model_kwargs['alm_threshold']=avg_thresholds
        return model_kwargs, noradid

    def run_dataset_average(self, model_type, model_kwargs):
        # Create a dict to store the current model metrics
        metrics = {
            'loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'f2': 0.0,
            'mttd': 0.0,
            'train_time': 0.0,
            'aucroc': 0.0,
            'threshold': 0.0
        }

        save_path = join(rootdir, 'results', 'models', model_type, f'{self.combi_index}.pt')

        for i in range(self.amount_of_entry):
            self.logger.debug(f"------------------------- Time series {i+1}/{self.amount_of_entry} -------------------------")

            # Split the data
            train_val_split_dataset = self.train_val_split_dataset
            train_data = train_val_split_dataset[i]['train_data']
            train_labels = train_val_split_dataset[i]['train_labels']
            val_data = train_val_split_dataset[i]['val_data']
            val_labels = train_val_split_dataset[i]['val_labels']
            
            test_split_dataset = self.test_split_dataset
            test_data = test_split_dataset[i]['test_data']
            test_labels = test_split_dataset[i]['test_labels']
            
            # Add the alm_threshold adapted to the current data set
            model_kwargs, noradId = self.threshold_adapt(model_kwargs, train_data)
            self.logger.debug(f'alm_threshold: {pprint.pformat(model_kwargs['alm_threshold'])} | train_start_date: {datetime.fromtimestamp(train_data.t0)}')
            
            # Set the model with current parameters
            model = self.model_types[model_type](config=self.model_types[model_type].config_class(**model_kwargs))

            # Evaluate the model
            if self.evaluate:
                model._build_empty_model(train_data.shape[0])
                _, _, _, _, loss = self.load_checkpoint(model.model, None, save_path)
                scores, precision, recall, f1, f2, mttd = self.eval(model, test_data, test_labels)
                train_time=0.0
            else:
                precision, recall, f1, f2, mttd, train_time, threshold, aucroc = self.train_eval(model, train_data, val_data, val_labels, test_data, test_labels, model_type, model_kwargs, noradId, plot=False)
                loss = model.total_loss.item()
                
            # Increment metrics
            for metric, value in zip(metrics, [loss, precision, recall, f1, f2, mttd, train_time, aucroc, threshold]):
                metrics[metric] += value

        # Average the metrics over the amount of data entries
        for metric in metrics:
            metrics[metric] /= self.amount_of_entry

        # Save model
        if not self.evaluate:
            _=0
            #save_checkpoint(model=model.model, kargs=model_kwargs, optimizer=model.optimizer, epoch=model.num_epochs, loss=loss, save_path=save_path)
        
        return metrics

    def hyperparam_opti(self, model_type):
        if self.log_wandb:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="MerlionAnomalyDetector",
                resume='allow',
                id=f'{model_type}-{run_id}',
                
                # track hyperparameters and run metadata
                config={
                "model": model_type,
                "dataset": "residues"
                }
            )
        # Load hyperparams_dict from the JSON file
        with open(join(rootdir, 'conf', 'hyperparam_optimization.json'), 'r') as json_file:
            hyperparams_dict = json.load(json_file)
        
        hyperparams = hyperparams_dict[model_type]
        
        # Convert some lists to tuple
        if model_type == 'VAE':
            hyperparams['encoder_hidden_sizes'] = [tuple(param) for param in hyperparams['encoder_hidden_sizes']]
        if model_type == 'AutoEncoder':
            hyperparams['layer_sizes'] = [tuple(param) for param in hyperparams['layer_sizes']]
            
        # Generate all combinations of hyperparameters
        hyperparam_combinations = list(itertools.product(*hyperparams.values()))

        # Initialize the utility function
        best_utility_function = 0.0

        self.logger.warning(f'{model_type} hyperparameter optimization: {len(hyperparam_combinations)} combinations to test.')
        for i, combination in enumerate(tqdm(hyperparam_combinations), start=0):
            self.combi_index = i
            # Update the model current configuration from the hyperparameters list
            model_kwargs = dict(zip(hyperparams.keys(), combination))
            if model_type == 'VAE': # Add the 'decoder_hidden_sizes' hyperparameter, symetric of 'encoder_hidden_sizes'
                model_kwargs['decoder_hidden_sizes'] = model_kwargs['encoder_hidden_sizes'][::-1]

            self.logger.info(f'{i}/{len(hyperparam_combinations)}: {' | '.join(['{0}: {1}'.format(k, v) for k,v in model_kwargs.items()])}')

            # Train and evaluate the model
            metrics = self.run_dataset_average(model_type, model_kwargs)
            self.logger.debug(f'Metrics for:\n{model_kwargs}\n{pprint.pformat(metrics)}')

            # Compute the utility function average of the considered metrics
            utility_function_variables = [metrics["f2"], metrics['train_time']]
            utility_function = sum(utility_function_variables)/len(utility_function_variables)
            
            if utility_function > best_utility_function:
                best_utility_function = utility_function
                best_model = model_kwargs
                best_metrics = metrics
                print()
                self.logger.info(f'New best {model_type}:\n{pprint.pformat(best_model)}\nConfiguration;\n{pprint.pformat(best_metrics)}')
            
            if self.log_wandb:
                # Log to WandB the model and its metrics (without hyperparameters in tuple format, like MPL layer sizes)
                wandb.log({key: val for key, val in {**model_kwargs, **metrics}.items() if ((not isinstance(val, tuple)) and (not key=='alm_threshold'))})
            print()
        
        self.logger.warning(f'The best model is {best_model}\nWith metrics {best_metrics}')
        
        if self.log_wandb:
            # End the WandB run
            wandb.finish()

    def single_model_train(self, model_type, model_kwargs):
        self.logger.info(f'Starting training for best {model_type} with {model_kwargs}')

        # Train and evaluate the model
        for i in range(self.amount_of_entry):
            self.logger.debug(f"------------------------- Time series {i+1}/{self.amount_of_entry} -------------------------")
           
            # Split the data
            train_val_split_dataset = self.train_val_split_dataset
            train_data = train_val_split_dataset[i]['train_data']
            train_labels = train_val_split_dataset[i]['train_labels']
            val_data = train_val_split_dataset[i]['val_data']
            val_labels = train_val_split_dataset[i]['val_labels']
            
            # Add the alm_threshold adapted to the current data set
            _, noradId = self.threshold_adapt(model_kwargs, train_data)
            self.logger.debug(f'alm_threshold: {pprint.pformat(model_kwargs['alm_threshold'])} | train_start_date: {datetime.fromtimestamp(train_data.t0)}')
            
            if self.log_wandb:
                # start a new wandb run to track this script
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="MerlionAnomalyDetector",
                    resume='allow',
                    id=f'Best-{model_type}-{noradId}-{run_id}',
                    
                    # track hyperparameters and run metadata
                    config={
                    "model": model_type,
                    "hyperparameters": model_kwargs,
                    "dataset": f"residues_{noradId}",
                    "job_type": "best_model_training"
                    }
                )
            
            # Set the model with correct hyperparameters
            model = copy.deepcopy(self.model_types[model_type](config=self.model_types[model_type].config_class(**model_kwargs)))
            
            # Preprocess the train data
            train_data_pd = model.train_pre_process(train_data)
            train_data_pd = train_data.to_pd() if model._pandas_train else train_data
                       
            # Build the model
            model.model = model._build_model(train_data_pd.shape[1]).to(model.device)
            model.data_dim = train_data_pd.shape[1]
            
            # Setup the model
            model.setup_model(train_data=train_data_pd)
            bar = ProgressBar(total=model.num_epochs)

            # Train
            model.total_loss=0.0
            for epoch in range(model.num_epochs):
                self.logger.debug(f"----------------- Epoch {epoch}/{model.num_epochs} -----------------")

                # Check if the save directory exists, if not, create it
                save_path = join(rootdir, 'results', 'models', model_type, 'best', noradId, f'{epoch}.pt')
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                if not self.evaluate:
                    # Train the model for one epoch
                    loss = model._train_one_epoch()
                else:
                    # Load the model state at the current epoch
                    model.model, model.optimizer, model_kwargs, epoch, loss = self.load_checkpoint(model.model, model.optimizer, save_path)
                    self.logger.debug(f'Loaded model at epoch {epoch} with loss {loss}')

                scores, train_precision, train_recall, train_f1, train_f2, train_mttd = self.eval(model, train_data, train_labels, method='pa')
                scores, test_precision, test_recall, test_f1, test_f2, test_mttd = self.eval(model, val_data, val_labels, method='pa')
                metrics = {
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "train_f2": train_f2,
                    "train_mttd": train_mttd,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                    "test_f2": test_f2,
                    "test_mttd": test_mttd
                }
                self.logger.debug(f'Metrics:\n{pprint.pformat(metrics)}')
                if bar is not None:
                    bar.print(epoch, prefix="", suffix="Complete, Loss {:.4f}".format(loss))

                if not eval:
                    # Save the model state at the current epoch
                    save_checkpoint(model=model.model, kargs=model_kwargs, optimizer=model.optimizer, epoch=model.num_epochs, loss=loss, save_path=save_path)

                if self.log_wandb:
                    # Log to WandB the model and its metrics (without hyperparameters in tuple format, like MPL layer sizes)
                    wandb_output = metrics.copy()
                    wandb_output['loss'] = loss
                    wandb_output['epoch'] = epoch
                    wandb.log(wandb_output)
                print()

            if self.log_wandb:
                # End the WandB run
                wandb.finish()


best_models_kargs = {
    "AutoEncoder": {'hidden_size': 11,
                    'layer_sizes': (15, 10, 5),
                    'sequence_len': 1,
                    'lr': 0.01,
                    'batch_size': 512,
                    'num_epochs': 500},
    "VAE": {'encoder_hidden_sizes': (15, 10, 5),
            'decoder_hidden_sizes': (5, 10, 15),
            'latent_size': 11,
            'sequence_len': 1,
            'kld_weight': 1.0,
            'dropout_rate': 0.0,
            'num_eval_samples': 5,
            'lr': 0.0005,
            'batch_size': 1024,
            'num_epochs': 500
            },
    "LSTMED": {'hidden_size': 0,
                    'layer_sizes': (0, 0, 0),
                    'sequence_len': 0,
                    'lr': 0,
                    'batch_size': 0,
                    'num_epochs': 0}
    }


def main():
    run_type = 'hyperparam_opti'
    # run_type = 'train'
    
    for model_id in [2,2]:
        md = ManeuverDetection(dataset=None, log_wandb=False, evaluate=False, log=True, save_log=False)
        model_type = list(md.model_types.keys())[model_id]
        
        if run_type == 'train':
            # Train with given hyperparameters
            md.single_model_train(model_type=model_type, model_kwargs=best_models_kargs[model_type])
        
        elif run_type == 'hyperparam_opti':
            # Optimize hyperparameters
            md.hyperparam_opti(model_type)

if __name__ == "__main__":
    main()