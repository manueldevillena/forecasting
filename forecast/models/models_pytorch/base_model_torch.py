import logging
import numpy as np
import time
import torch

from abc import ABC
from torch.autograd import Variable

from forecast.core import FeatureCreation
from forecast.models import BaseModel


class BaseModelTorch(BaseModel, ABC):
    """
    Collection of methods used by all torch models.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__()
        for attr in ['size_output', 'num_layers', 'size_input', 'size_hidden',
                     'learning_rate', 'num_epochs', 'optimizer', 'criterion',
                     'X_train', 'y_train', 'X_test', 'y_test', 'X_scaled', 'y_scaled',
                     'X_scaler', 'y_scaler']:
            if attr not in features.config and attr not in features.features:
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))
            else:
                if attr in features.config:
                    setattr(self, attr, features.config[attr])
                else:
                    setattr(self, attr, features.features[attr])

        self.X_train_tensor = self._create_X_tensor(self.X_train)
        self.y_train_tensor = self._create_y_tensor(self.y_train)
        self.X_tensor = self._create_X_tensor(self.X_scaled)
        self.y_tensor = self._create_y_tensor(self.y_scaled)

    @staticmethod
    def _train(model):
        """
        Trains the pytorch model.
        Args:
            model: Model to be trained
        """
        tic = time.time()
        running_loss = 0.0
        for epoch in range(model.num_epochs):  # Iterate over number of epochs

            outputs = model.forward(model.X_train_tensor)  # forward pass
            model.optimizer.zero_grad()  # calculates the gradient, manually setting to 0

            # obtain the loss function
            loss = model.criterion(outputs, model.y_train_tensor)

            loss.backward()  # calculates the loss of the loss function

            model.optimizer.step()  # improve from loss, i.e., backpropagation

            running_loss += loss.item() * model.X_train_tensor.size(0)
            epoch_loss = running_loss / len(model.y_train_tensor)

            if epoch % 100 == 0:
                logging.info('-' * 20)
                logging.info('Epoch: {}/{}'.format(epoch, model.num_epochs - 1))
                logging.info("Loss: {:.4f}".format(epoch_loss))
                logging.info('-' * 20)
                # print("Epoch: {}, loss: {:4f}".format(epoch, loss.item()))
            elif epoch == model.num_epochs - 1:
                logging.info('-' * 20)
                logging.info('-' * 20)
                logging.info("Final loss: {:4f}".format(loss.item()))

        tac = time.time() - tic
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(tac // 60, tac % 60))

    def _predict(self, model, x_test: object = None) -> np.array:
        """
        Predicts using the trained model.
        Args:
            model: Model used to predict.
        """
        if x_test is not None: # TODO: take care of this case
            pass
        else:
            prediction_torch_scaled = model.forward(self.X_tensor)
            prediction_numpy_scaled = prediction_torch_scaled.data.numpy()
            true_values_numpy_scaled = self.y_tensor.data.numpy()

            prediction_numpy = self.y_scaler.inverse_transform(prediction_numpy_scaled)
            true_values_numpy = self.y_scaler.inverse_transform(true_values_numpy_scaled)

        data_to_plot = {
            'prediction_numpy': prediction_numpy,
            'true_values_numpy': true_values_numpy
        }
        return data_to_plot

    @staticmethod
    def _create_X_tensor(X_array: np.array) -> torch.Tensor:
        """
        Creates a torch tensor from an array of inputs.
        Args:
            array: Input array to be converted to a tensor

        Returns:
            Inputs torch tensor
        """
        X_tensor = Variable(torch.Tensor(X_array))
        X_tensor_reshaped = torch.reshape(
            X_tensor,
            (
                X_tensor.shape[0], 1, X_tensor.shape[1]
            )
        )
        return X_tensor_reshaped

    @staticmethod
    def _create_y_tensor(y_array: np.array) -> torch.Tensor:
        """
        Creates a torch tensor from an array of targets.
        Args:
            array: Target array to be converted to a tensor

        Returns:
            Target torch tensor
        """
        y_tensor = Variable(torch.Tensor(y_array))
        return y_tensor