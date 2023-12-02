#!pip install pandas matplotlib

import torch

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

from merlion.models.anomaly.autoencoder import AutoEncoder
from merlion.models.anomaly.vae import VAE
from merlion.models.anomaly.isolation_forest import IsolationForest
from merlion.evaluate.anomaly import TSADMetric
from ts_datasets.anomaly import CustomAnomalyDataset
from merlion.utils import TimeSeries


rootdir = dirname(abspath(__file__))
run_id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

## Create a logger
logger = logging.getLogger('ManeuverDetectionLogger')
logger.setLevel(logging.DEBUG)
# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler() #sys.stdout
console_handler.setLevel(logging.INFO)
# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler(join(rootdir, 'results', 'logs', f'{run_id}.log'))
file_handler.setLevel(logging.DEBUG)
# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - (%(module)s:%(lineno)d): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_train_test(dataset, entry, normalize=False):
    time_series, metadata = dataset[entry]

    # time_series = time_series.rename(columns={'INCLINATION': '0', 
    #                             'RA_OF_ASC_NODE': '1', 
    #                             'ECCENTRICITY': '2', 
    #                             'ARG_OF_PERICENTER': '3', 
    #                             'MEAN_ANOMALY': '4', 
    #                             'MEAN_MOTION': '5'
    #                             })

    train_data = time_series[metadata.trainval]
    test_data = time_series[~metadata.trainval]
    test_labels = metadata.anomaly[~metadata.trainval]

    # Normalize each row
    #TODO: Normalize by column. By row doesn't make sense.
    if normalize:
        train_data = train_data.apply(lambda x: x/abs(x).max(), axis=1) # Divide each row by the highest feature in absolute
        test_data = test_data.apply(lambda x: x/abs(x).max(), axis=1) # Divide each row by the highest feature in absolute

    train_data = TimeSeries.from_pd(train_data)
    test_data = TimeSeries.from_pd(test_data)
    test_labels = TimeSeries.from_pd(test_labels)
    
    return train_data, test_data, test_labels

def crop_dataset(split_dataset):
    for key in split_dataset[0].keys():
        series_lengths = [len(split_dataset[i][key]) for i in range(len(split_dataset))]
        min_series_length = min(series_lengths)
        max_series_length = max(series_lengths)
        logger.debug(f'{key} lengths: {series_lengths} | Smallest: {min_series_length}: {series_lengths.index(min_series_length)} | Longest: {max_series_length}: {series_lengths.index(max_series_length)}')
        logger.info(f'Cropping {key} to a length of {min_series_length}')
        for i in range(len(split_dataset)):
            split_dataset[i][key] = TimeSeries.from_pd(TimeSeries.to_pd(split_dataset[i][key]).head(min_series_length))

    return split_dataset    

def train(model, train_data):
    train_start = datetime.now()
    ## Train
    train_scores = model.train(train_data)
    train_time = datetime.now()-train_start
    
    return train_time, train_scores

def eval(model, test_data, test_labels):
    ## Evaluate
    labels = model.get_anomaly_label(test_data)
    precision = TSADMetric.PointAdjustedPrecision.value(ground_truth=test_labels, predict=labels)
    recall = TSADMetric.PointAdjustedRecall.value(ground_truth=test_labels, predict=labels)
    f1 = TSADMetric.PointAdjustedF1.value(ground_truth=test_labels, predict=labels)
    f2 = 5*(precision*recall)/(4*precision+recall) if precision>0.0 and recall>0.0 else 0
    mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=labels) # Corresponds to the mean time between the first point classified 
                                                                                       # as anomalous and the start of an actual anomaly window
                                                                                       
    return precision, recall, f1, f2, mttd

def train_eval(model, train_data, test_data, test_labels):
    train_time, train_scores = train(model, train_data)
    precision, recall, f1, f2, mttd = eval(model, test_data, test_labels)

    return precision, recall, f1, f2, float(mttd.total_seconds()), float(train_time.total_seconds())

def save_checkpoint(model, kargs, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'kargs': kargs
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    kargs = checkpoint['kargs']
    return model, kargs, optimizer, epoch, loss


class ManeuverDetection():

    def __init__(self, dataset, log_wandb=False, eval=True):
        logger.info(f'Initializing ManeuverDetection: log_wandb: {log_wandb} - eval:{eval}')
        self.dataset_dir = join(rootdir, 'data', 'residues')
        self.model_types = {
            'AutoEncoder': type(AutoEncoder(config=AutoEncoder.config_class())),
            'VAE': type(VAE(config=VAE.config_class())),
            'IsolationForest': type(IsolationForest(config=IsolationForest.config_class()))
            }
        
        self.eval=eval
        self.log_wandb=log_wandb
        
        if dataset is None:
            logger.info(f'Fetching dataset: {self.dataset_dir}')
            self.dataset = CustomAnomalyDataset(
                rootdir=self.dataset_dir,   # where the data is stored
                test_frac=0.70              # use 100*test_frac % of each time series for testing.
                                            # overridden if the column `trainval` is in the actual CSV.
                #time_unit="s",             # the timestamp column (automatically detected) is in units of seconds
                #assume_no_anomaly=True     # if a CSV doesn't have the "anomaly" column, assume it has no anomalies
            )
        self.amount_of_entry = len(self.dataset)
        
        # Split the dataset
        self.split_dataset = crop_dataset({i: {'train_data': train_data, 'test_data': test_data, 'test_labels': test_labels}
            for i, (train_data, test_data, test_labels) in enumerate(map(lambda i: get_train_test(self.dataset, i), range(self.amount_of_entry)))})
        
        # Load thresholds from the JSON file
        with open(join(rootdir, 'conf', 'alm_thresholds.json'), 'r') as json_file:
            self.thresholds = json.load(json_file)


    def threshold_adapt(self, model_kwargs, train_data):
        train_start_date = datetime.fromtimestamp(train_data.t0)
        tmp_thresholds = [value['alm_threshold'] for key, value in self.thresholds.items()]
        avg_thresholds = sum(tmp_thresholds) / len(tmp_thresholds)
        for key, value in self.thresholds.items():
            start_date = datetime.strptime(value['start_date'], "%Y-%m-%d")
            #logger.debug(f'Compare dict {start_date} with real {train_start_date}')
            error_margin = timedelta(1)
            if (train_start_date > start_date - error_margin) and (train_start_date < start_date + error_margin):
                #logger.debug(f'Found correspondance.')
                model_kwargs['alm_threshold']=value['alm_threshold']
                break
            else:
                #logger.debug(f'No correspondance found.')
                model_kwargs['alm_threshold']=avg_thresholds
        return model_kwargs

    def run_dataset_average(self, model_type, model_kwargs):   
        # Create a dict to store the current model metrics
        current_metrics = {
            'loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'f2': 0.0,
            'mttd': 0.0,
            'train_time': 0.0
        }

        save_path = join(rootdir, 'results', 'models', model_type, f'{self.combi_index}.pt')

        for i in range(self.amount_of_entry):
            logger.debug(f"------------------------- Time series {i+1}/{self.amount_of_entry} -------------------------")

            # Split the data
            split_dataset = self.split_dataset
            train_data = split_dataset[i]['train_data']
            test_data = split_dataset[i]['test_data']
            test_labels = split_dataset[i]['test_labels']
            
            # Add the alm_threshold adapted to the current data set
            model_kwargs = self.threshold_adapt(model_kwargs, train_data)
            logger.debug(f'alm_threshold: {pprint.pformat(model_kwargs['alm_threshold'])} | train_start_date: {datetime.fromtimestamp(train_data.t0)}')
            
            # Set the model with current parameters
            model = self.model_types[model_type](config=self.model_types[model_type].config_class(**model_kwargs))

            # Evaluate the model
            if self.eval:
                model._build_empty_model(train_data.shape[0])
                _, _, _, _, loss = load_checkpoint(model.model, None, save_path)
                precision, recall, f1, f2, mttd = eval(model, test_data, test_labels)
                train_time=0.0
            else:
                precision, recall, f1, f2, mttd, train_time = train_eval(model, train_data, test_data, test_labels)
                loss = model.total_loss.item()

            # Increment metrics
            for metric, value in zip(current_metrics, [loss, precision, recall, f1, f2, mttd, train_time]):
                current_metrics[metric] += value

        # Average the metrics over the amount of data entries
        for metric in current_metrics:
            current_metrics[metric] /= self.amount_of_entry

        # Save model
        if not self.eval:
            save_checkpoint(model=model.model, kargs=model_kwargs, optimizer=model.optimizer, epoch=model.num_epochs, loss=loss, save_path=save_path)
        
        return current_metrics


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
        with open(join(rootdir, 'conf', 'maneuver_detection.json'), 'r') as json_file:
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
        
        for i, combination in enumerate(tqdm(hyperparam_combinations), start=1):
            self.combi_index = i
            # Update the model current configuration from the hyperparameters list
            model_kwargs = dict(zip(hyperparams.keys(), combination))
            if model_type == 'VAE': # Add the 'decoder_hidden_sizes' hyperparameter, symetric of 'encoder_hidden_sizes'
                model_kwargs['decoder_hidden_sizes'] = model_kwargs['encoder_hidden_sizes'][::-1]
            
            # Train and evaluate the model
            metrics = self.run_dataset_average(model_type, model_kwargs)
            logger.debug(f'Metrics for:\n{model_kwargs}\n{pprint.pformat(metrics)}')

            # Compute the utility function average of the considered metrics
            utility_function_variables = [metrics["f2"], metrics['train_time']]
            utility_function = sum(utility_function_variables)/len(utility_function_variables)
            
            if utility_function > best_utility_function:
                best_utility_function = utility_function
                best_model = model_kwargs
                best_metrics = metrics
                print()
                logger.info(f'New best {model_type}:\n{pprint.pformat(best_model)}\nConfiguration;\n{pprint.pformat(best_metrics)}')
            
            if self.log_wandb:
                # Log to WandB the model and its metrics (without hyperparameters in tuple format, like MPL layer sizes)
                wandb.log({key: val for key, val in {**model_kwargs, **metrics}.items() if ((not isinstance(val, tuple)) and (not key=='alm_threshold'))})
            print()
        
        logger.warning(f'The best model is {best_model}\nWith metrics {best_metrics}')
        
        if self.log_wandb:
            # End the WandB run
            wandb.finish()
            
md = ManeuverDetection(dataset=None, log_wandb=True, eval=False)

model_id = 0

model_type = list(md.model_types.keys())[model_id]

md.hyperparam_opti(model_type)