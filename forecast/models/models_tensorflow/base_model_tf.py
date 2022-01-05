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
    def __init__(self, model_hyperparameters: dict):
        """
        Constructor.
        """
        super().__init__()
        for attr in ['batch_size', 'learning_rate', 'num_epochs', 'optimizer', 'criterion', 'size_output']:
            if attr not in model_hyperparameters:
                raise KeyError(f'Attribute "{attr}" is mandatory in the configuration file.')
            else:
                setattr(self, attr, model_hyperparameters[attr])

    @staticmethod
    def _train(model, dataset):
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
            dataset['X_train_scaled'],
            dataset["y_train_scaled"],
            batch_size=model.batch_size,
            epochs=model.num_epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(dataset['X_val_scaled'], dataset['y_val_scaled']),
            callbacks=[callback]
        )

        tac = time.perf_counter() - tic
        logging.info(f'Training complete in {tac//60:.0f}m {tac%60:.0f}s')

    def _predict(self, model, dataset):
        """
        Predicts using the trained model.
        Args:
            model: Model used to predict.
        """

        X_tensor = self._create_tensor(dataset['X_test_scaled'])
        predicted_values_tf_scaled = model(X_tensor, training=False)
        predicted_values_numpy_scaled = predicted_values_tf_scaled.numpy()

        predicted_values = dataset['y_scaler'].inverse_transform(predicted_values_numpy_scaled)

        return predicted_values

    def _create_tensor(self, input_array: np.array):
        return tf.convert_to_tensor(input_array, dtype=tf.float32)
