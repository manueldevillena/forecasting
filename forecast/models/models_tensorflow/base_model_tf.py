import logging
import numpy as np
import time

from abc import ABC
import abc
from forecast.core import FeatureCreation
from forecast.models import BaseModel
import tensorflow as tf

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
                raise KeyError(f'Attribute "{attr}" is mandatory in the configuration file.')
            else:
                if attr in features.config:
                    setattr(self, attr, features.config[attr])
                else:
                    setattr(self, attr, features.features[attr])

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_scaled, self.y_train_scaled)) \
                                .shuffle(buffer_size=1024).batch(32)

    @abc.abstractmethod
    @tf.function
    def train_step(self, x, y):
        pass

    @staticmethod
    def _train(model):
        """
        Trains the tensorflow model.
        Args:
            model: Model to be trained
        """
        tic = time.time()
        for epoch in range(model.num_epochs):
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(model.train_dataset):
                loss_value = model.train_step(x_batch_train, y_batch_train)

            # Log every 10 epochs.
            if epoch % 10 == 0:
                print(
                    "Training loss for epoch %d: %.4f"
                    % (epoch, float(loss_value))
                )

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
            X_tensor = self._create_tensor(X_scaled)
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