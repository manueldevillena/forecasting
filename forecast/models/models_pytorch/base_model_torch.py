import logging
import numpy as np
import time
import torch

from abc import ABC
from torch.autograd import Variable

from forecast.core import FeatureCreation
from forecast.models import BaseModel
from forecast.utils import infer_activation


class BaseModelTorch(BaseModel, ABC):
    """
    Collection of methods used by all torch models.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__()
        for attr in ['size_output', 'num_layers_lstm', 'num_layers_linear', 'size_input', 'size_hidden',
                     'activation_function', 'learning_rate', 'num_epochs', 'optimizer', 'criterion',
                     'X', 'y', 'X_train_scaled', 'y_train_scaled', 'X_scaler', 'y_scaler']:
            if attr not in features.config and attr not in features.features:
                raise KeyError(f'Attribute "{attr}" is mandatory in the configuration file.')
            else:
                if attr in features.config:
                    setattr(self, attr, features.config[attr])
                else:
                    setattr(self, attr, features.features[attr])

        self.X_train_tensor = self._create_X_tensor(self.X_train_scaled)
        self.y_train_tensor = self._create_y_tensor(self.y_train_scaled)
        self.X_tensor = self._create_X_tensor(self.X)
        # self.y_tensor = self._create_y_tensor(self.y_scaled)

        self.activation = infer_activation(self.activation_function)

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
                logging.info(f'Epoch: {epoch}/{model.num_epochs-1}')
                logging.info(f'Loss: {epoch_loss:.4f}')
                logging.info('-' * 20)
                # print(f'Epoch: {epoch}, loss: {loss.item():4f}'))
            elif epoch == model.num_epochs - 1:
                logging.info('-' * 20)
                logging.info('-' * 20)
                logging.info(f'Final loss: {loss.item():4f}')

        tac = time.time() - tic
        logging.info(f'Training complete in {tac//60:.0f}m {tac%60:.0f}s')

    def _predict(self, model, x_test: object = None) -> dict:
        """
        Predicts using the trained model.
        Args:
            model: Model used to predict.
        """
        if x_test is not None:  # TODO: take care of this case
            pass
        else:
            X_scaled = self.X_scaler.fit_transform(self.X)
            X_tensor = self._create_X_tensor(X_scaled)
            predicted_values_torch_scaled = model.forward(X_tensor)
            predicted_values_numpy_scaled = predicted_values_torch_scaled.data.numpy()

            predicted_values_numpy = self.y_scaler.inverse_transform(predicted_values_numpy_scaled)

        data_to_plot = {
            'predicted_values_numpy': predicted_values_numpy,
            'actual_values_numpy': self.y
        }
        return data_to_plot

    def create_linear_net(self):
        """
        Creates the layers of the linear network.
        """
        net_layers = list()
        input_size = self.size_hidden
        for n_neurons in self.num_layers_linear:
            net_layers.append(torch.nn.Linear(input_size, n_neurons))
            net_layers.append(self.activation.__class__())

            input_size = n_neurons

        net_layers.append(torch.nn.Linear(n_neurons, self.size_output))
        net_layers.append(self.activation.__class__())

        return net_layers

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
