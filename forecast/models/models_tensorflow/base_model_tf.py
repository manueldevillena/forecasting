import logging
import numpy as np
import time

import abc
from forecast.core import FeatureCreation
from forecast.models import BaseModel
import tensorflow as tf


class BaseModelTF(BaseModel, abc.ABC):
    """
    Collection of methods used by all tensorflow (with keras) models.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__()
        for attr in ['size_output', 'num_layers_lstm', 'num_layers_linear', 'size_input', 'size_hidden', 'batch_size',
                     'activation_function', 'learning_rate', 'num_epochs', 'optimizer', 'criterion',
                     'X', 'y', 'X_train_scaled', 'y_train_scaled', 'X_scaler', 'y_scaler', 'X_val_scaled', 'y_val_scaled']:
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
        Trains the tensorflow model.
        Args:
            model: Model to be trained
        """
        tic = time.perf_counter()
        model.compile(
            optimizer=model.optimizer,  # Optimizer
            # Loss function to minimize
            loss=model.criterion,
            # List of metrics to monitor
            # metrics=[],
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            model.X_train_scaled,
            model.y_train_scaled,
            batch_size=model.batch_size,
            epochs=model.num_epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(model.X_val_scaled, model.y_val_scaled),
            callbacks=[callback]
        )

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
            X_tensor = self._create_tensor(self.X)
            predicted_values_tf_scaled = model(X_tensor, training=False)
            predicted_values_numpy_scaled = predicted_values_tf_scaled.numpy()

            predicted_values_numpy = self.y_scaler.inverse_transform(predicted_values_numpy_scaled)

        data_to_plot = {
            'predicted_values_numpy': predicted_values_numpy,
            'actual_values_numpy': self.y
        }
        return data_to_plot

    def _create_tensor(self, input_array: np.array):
        return tf.convert_to_tensor(input_array, dtype=tf.float32)
