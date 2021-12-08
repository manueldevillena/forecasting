import logging
import numpy as np
import time

from abc import ABC
from forecast.core import FeatureCreation
from forecast.models import BaseModel


class BaseModelSKLearn(BaseModel, ABC):
    """
    Collection of methods used by all torch models.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__()
        for attr in ['size_output', 'num_layers_lstm', 'num_layers_linear', 'size_input', 'size_hidden', 'batch_size',
                     'activation_function', 'learning_rate', 'num_epochs', 'optimizer', 'criterion',
                     'X', 'y', 'X_train_scaled', 'y_train_scaled', 'X_scaler', 'y_scaler',
                     'estimators', 'criterion_trees', 'max_depth', 'jobs']:
            if attr not in features.config and attr not in features.features:
                raise KeyError(f'Attribute "{attr}" is mandatory in the configuration file.')
            else:
                if attr in features.config:
                    setattr(self, attr, features.config[attr])
                else:
                    setattr(self, attr, features.features[attr])

    @staticmethod
    def _train(model):
        """
        Trains the pytorch model.
        Args:
            model: Model to be trained
        """
        tic = time.perf_counter()
        model.model.fit(model.X_train_scaled, model.y_train_scaled)

        tac = time.perf_counter() - tic
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
            predicted_values_scaled = model.model.predict(X_scaled)

            predicted_values = self.y_scaler.inverse_transform(predicted_values_scaled)

        data_to_plot = {
            'predicted_values_numpy': predicted_values,
            'actual_values_numpy': self.y
        }
        return data_to_plot
