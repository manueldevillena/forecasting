import logging
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class ParsingException(Exception):
    def __init__(self, indexes: list):
        super().__init__()
        self.indexes = indexes

    def __str__(self):
        return "Invalid values at indexes:\n\t" + '\n\t'.join(map(str, self.indexes))


def read_inputs(data_path: str) -> pd.DataFrame:
    """
    Reads CSV file returning an object.

    :param data_path: Path to the data to read.
    :return Dataframe with the read data.
    """
    df = pd.read_csv(data_path, header=0, index_col=0, parse_dates=True, infer_datetime_format=True, dtype=float)
    null_loc, _ = np.where(df.isna())
    if len(null_loc) != 0:
        raise ParsingException(indexes=df.index[null_loc])
    return df


def read_config(inputs_path: str) -> dict:
    """
    Reads YML file with inputs.
    :param inputs_path: Path to the inputs file.
    :return: Dictionary with the read data.
    """
    with open(inputs_path) as infile:
        data = yaml.load(infile, Loader=Loader)
    return data


def infer_activation(activation: str, mode: str = 'pytorch'):
    """
    Maps the activation function to the given string.
    Args:
        activation: String with activation ('relu', 'sigmoid', or 'tanh')

    Returns:
        Activation function
    """
    if mode == 'pytorch':
        activations = {
            'relu': torch.nn.ReLU(),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh()
        }
    elif mode == 'tensorflow':
        activations = {
            'relu': tf.keras.layers.Activation('relu'),
            'sigmoid': tf.keras.layers.Activation('sigmoid'),
            'tanh': tf.keras.layers.Activation('tanh')
        }

    return activations[activation.lower()]


def infer_optimizer(model: object, mode: str = 'pytorch'):
    """
    Maps the optimizer string given to the appropriate optimizer.
    Args:
        mode:
        model: Model with info of the optimizer to use

    Returns:
        Optimizer to use
    """
    if mode == 'pytorch':
        optimizers = {
            'adam': torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        }
    elif mode == 'tensorflow':
        optimizers = {
            'adam': tf.optimizers.Adam(learning_rate=model.learning_rate)
        }
    return optimizers[model.optimizer]


def infer_criterion(criterion: str, mode: str = 'pytorch'):
    """
    Maps the criterion string given to the appropriate criterion.
    Args:
        mode:
        model: Model with info of the criterion to use

    Returns:
        Criterion to use
    """
    if mode == 'pytorch':
        criteria = {
            'mse': torch.nn.MSELoss()
        }
    elif mode == 'tensorflow':
        criteria = {
            'mse': tf.losses.MeanSquaredError()
        }
    return criteria[criterion]


def infer_scaler(scaler):
    """
    Maps the scaler sring given to the appropriate scaler.
    Args:
        scaler:

    Returns:
        Scaler to use
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    return scalers[scaler]


def numpy_to_torch(array):
    """
    Transforms a numpy array into a torch tensor.
    Args:
        array: Given numpy array

    Returns:
        Torch tensor
    """
    return torch.tensor(array.astype(np.float), dtype=torch.float32)


def create_linear_network(input_size, layers, activation):
    """
    Creates a linear neural network.
    Args:
        input_size:
        layers:
        act_fun:

    Returns:
        Network layers
    """
    net_layers = list()
    for n_neurons in layers:
        # linear layers
        net_layers.append(torch.nn.Linear(input_size, n_neurons))
        net_layers.append(activation.__class__())
        input_size = n_neurons

    return net_layers


def set_up_logger(path, log_in_file=True):
    """
    Sets up the logger.
    Args:
        path:
        log_in_file:

    Returns:

    """
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rootLogger = logging.getLogger()

    if rootLogger.hasHandlers():
        for hdlr in rootLogger.handlers[:]:  # remove all old handlers
            rootLogger.removeHandler(hdlr)
    if log_in_file:
        fileHandler = logging.FileHandler(f'{path}/logfile.log')
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    return rootLogger


def compute_metrics(y, y_prime):
    """
    Computes metrics MAE, MAPE, MSE.
    Args:
        y: actual targets
        y_prime: predicted targets

    Returns:
        Dictionary with metrics
    """
    return {
        'MAE': mean_absolute_error(y, y_prime),
        'MAPE': mean_absolute_percentage_error(y, y_prime),
        'MSE': mean_squared_error(y, y_prime)
    }