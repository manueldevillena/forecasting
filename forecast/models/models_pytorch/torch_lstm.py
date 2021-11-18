import logging
import time
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
                     'learning_rate', 'num_epochs', 'optimizer', 'criterion',
                     'X_train', 'y_train', 'X_test', 'y_test']:
            if attr not in features.config and attr not in features.features:
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))
            else:
                if attr in features.config:
                    setattr(self, attr, features.config[attr])
                else:
                    setattr(self, attr, features.features[attr])

        self.X_tensor, self.y_tensor = self.create_tensors(self.X_train, self.y_train)
        self.seq_length = self.X_tensor.shape[1]

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

    def train(self):
        tic = time.time()
        running_loss = 0.0
        for epoch in range(self.num_epochs):  # Iterate over number of epochs

            outputs = self.forward(self.X_tensor)  # forward pass
            self.optimizer.zero_grad()  # calculates the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs, self.y_tensor)

            loss.backward()  # calculates the loss of the loss function

            self.optimizer.step()  # improve from loss, i.e., backpropagation

            running_loss += loss.item() * self.X_tensor.size(0)
            epoch_loss = running_loss / len(self.y_tensor)

            if epoch % 100 == 0:
                logging.info('-' * 20)
                logging.info('Epoch: {}/{}'.format(epoch, self.num_epochs - 1))
                logging.info("Loss: {:.4f}".format(epoch_loss))
                logging.info('-' * 20)
                # print("Epoch: {}, loss: {:4f}".format(epoch, loss.item()))
            elif epoch == self.num_epochs - 1:
                logging.info('-' * 20)
                logging.info('-' * 20)
                logging.info("Final loss: {:4f}".format(loss.item()))

        tac = time.time() - tic
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(tac // 60, tac % 60))

    def predict(self):
        pass