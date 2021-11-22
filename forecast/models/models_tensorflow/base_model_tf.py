import logging
import numpy as np
import time

from abc import ABC

from forecast.core import FeatureCreation
from forecast.models import BaseModel
from forecast.utils import infer_activation


class BaseModelTF(BaseModel, ABC):
    """
    Collection of methods used by all tensorflow (with keras) models.
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
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))
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
        tac = time.time() - tic
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(tac // 60, tac % 60))

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
        pass

    @staticmethod
    def _create_X_tensor(X_array: np.array):
        """
        Creates a torch tensor from an array of inputs.
        Args:
            array: Input array to be converted to a tensor

        Returns:
            Inputs torch tensor
        """
        pass

    @staticmethod
    def _create_y_tensor(y_array: np.array):
        """
        Creates a torch tensor from an array of targets.
        Args:
            array: Target array to be converted to a tensor

        Returns:
            Target torch tensor
        """
        pass