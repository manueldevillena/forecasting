import numpy as np
import torch

from forecast.models.models_pytorch import TorchNeuralNetwork
from forecast.utils import create_linear_network


class FeedForwardRegressor(torch.nn.Module):
    """
    Feed forward neural network used for regression based on pytorch network.
    """
    def __init__(self, input_size: int, output_size: int, layers: tuple, activation: object):
        """
        Constructor.

        :param input_size: Number of features per sample
        :param output_size: Number of targets
        :param layers: Shape of the net
        :param activation: Activation function applied to each layer
        """
        super(FeedForwardRegressor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.activation = activation

        self.net_layers = create_linear_network(self.input_size, self.layers, self.activation)
        self._output_layer()
        self.net = torch.nn.Sequential(*self.net_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor state through the network.

        :param state: Input tensor
        :return: Output tensor
        """
        x = self.net(state)

        return x.view(-1, self.output_size)

    def _output_layer(self):
        """
        Configuration of the output layer.
        """
        self.net_layers.append(torch.nn.Linear(self.layers[-1], self.output_size))
        self.net_layers.append(self.activation.__class__())


class FeedForwardRegression(TorchNeuralNetwork):
    """
    Implementation of Pytorch feed forward network (regression) based on the pytorch parent neural network.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, input_scaler: str = None, target_scaler: str = None,
                 percentage_train: float = 0.7, batch_size: int = 24, num_epochs: int = 25, activation: str = 'relu',
                 device: str = 'cpu', lr: float = 0.01, layers: tuple = (10,)):
        """
        Constructor.
        """
        super().__init__(x, y, input_scaler, target_scaler, percentage_train, batch_size, num_epochs, activation,
                         device, lr)
        self.input_size = np.shape(x)[1]
        self.output_size = np.shape(y)[1]
        self.model = FeedForwardRegressor(self.input_size, self.output_size, layers, self.activation)
        self.criterion = torch.nn.MSELoss()
