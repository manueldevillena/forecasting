import torch

from forecast.driver import System
from forecast.core import FeatureCreation
from forecast.utils import infer_optimizer, infer_criterion
from forecast.models.models_pytorch import TorchLSTM


class TorchDriver(System):
    """
    Concrete driver for pytorch models.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.

        Args:
            model: Torch model instance
        """
        for attr in ['size_output', 'num_layers', 'size_input', 'size_hidden',
                     'learning_rate', 'num_epochs', 'optimizer', 'criterion']:
            try:
                setattr(self, attr, features.config[attr])
            except KeyError:
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))

        # self.optimizer = infer_optimizer(self.model)
        # self.criterion = infer_criterion(self.model)

    def train(self, X: torch.Tensor, y: torch.Tensor):
        for epoch in range(self.model.num_epochs):
            outputs = self.model.forward(X)  # forward pass
            self.optimizer.zero_grad()  # calculates the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs, y)

            loss.backward()  # calculates the loss of the loss function

            self.optimizer.step()  # improve from loss, i.e., backpropagation
            if epoch % 10 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def predict(self):
        pass