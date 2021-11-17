import torch
import torch.nn as nn

from torch.autograd import Variable

from forecast.core import FeatureCreation
from forecast.driver import System
from forecast.utils import infer_optimizer, infer_criterion


class TorchLSTM(nn.Module, System):
    """
    Basic LSTM (RNN) for day-ahead timeseries forecasting.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        Args:
            features: Object with appropriate configuration files.
        """
        super().__init__()
        for attr in ['size_output', 'num_layers', 'size_input', 'size_hidden',
                     'learning_rate', 'num_epochs', 'optimizer', 'criterion']:
            try:
                setattr(self, attr, features.config[attr])
            except KeyError:
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))

        self.lstm = nn.LSTM(input_size=self.size_input, hidden_size=self.size_hidden, num_layers=self.num_layers,
                            batch_first=True)
        self.fc_1 = nn.Linear(self.size_hidden, 128)
        self.fc = nn.Linear(128, self.size_output)
        self.relu = nn.ReLU()

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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.size_hidden))  # hidden state initialise
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.size_hidden))  # internal state initialise
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.size_hidden)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output

        return out

    def train(self, X: torch.Tensor, y: torch.Tensor):
        for epoch in range(self.num_epochs):
            outputs = self.forward(X)  # forward pass
            self.optimizer.zero_grad()  # calculates the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs, y)

            loss.backward()  # calculates the loss of the loss function

            self.optimizer.step()  # improve from loss, i.e., backpropagation
            if epoch % 10 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def predict(self):
        pass