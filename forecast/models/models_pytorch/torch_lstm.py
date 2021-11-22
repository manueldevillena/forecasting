import logging
import time
import torch
import torch.nn as nn

from torch.autograd import Variable

from forecast.core import FeatureCreation
from forecast.models.models_pytorch import BaseModelTorch
from forecast.utils import infer_optimizer, infer_criterion


class TorchLSTM(nn.Module, BaseModelTorch):
    """
    Basic LSTM (RNN) for day-ahead timeseries forecasting.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        Args:
            features: Object with appropriate configuration files.
        """
        nn.Module.__init__(self)
        BaseModelTorch.__init__(self, features)

        self.seq_length = self.X_train_tensor.shape[1]

        self.lstm = nn.LSTM(input_size=self.size_input, hidden_size=self.size_hidden, num_layers=self.num_layers_lstm,
                            batch_first=True)
        # self.fc_1 = nn.Linear(self.size_hidden, 128)
        # self.fc = nn.Linear(128, self.size_output)
        # self.relu = nn.ReLU()
        self.linear_layers = self.create_linear_net()
        self.net = nn.Sequential(*self.linear_layers)

        self.optimizer = infer_optimizer(self)
        self.criterion = infer_criterion(self.criterion)

    def forward(self, x):
        """
        Forward pass through the neural network.
        Args:
            x: Inputs (features) of the neural network.

        Returns:
            Outputs of the neural network pass.
        """
        h_0 = Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.size_hidden))  # hidden state initialise
        c_0 = Variable(torch.zeros(self.num_layers_lstm, x.size(0), self.size_hidden))  # internal state initialise
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.size_hidden)  # reshaping the data for Dense layer next
        out = self.net(output[:, -1, :])
        # out = self.relu(hn)
        # out = self.fc_1(out)  # first Dense
        # out = self.relu(out)  # relu
        # out = self.fc(out)  # Final Output

        return out

    def train(self):
        """
        Trains.
        """
        super()._train(self)

    def predict(self) -> dict:
        """
        Predicts.
        """
        return super()._predict(self)
