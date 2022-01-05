from forecast.core import FeatureCreation
from forecast.models.models_tensorflow import BaseModelTF
from forecast.utils import infer_optimizer, infer_criterion
import tensorflow as tf
import keras.layers as layers


class TFFeedForward(tf.keras.Model, BaseModelTF):
    """
    Feed forward network for day-ahead timeseries forecasting.
    """
    def __init__(self, model_hyperparameters: dict):
        """
        Constructor.
        Args:
            features: Object with appropriate configuration files.
        """
        tf.keras.Model.__init__(self)
        BaseModelTF.__init__(self, model_hyperparameters)

        self.output_projection = layers.Dense(self.size_output)
        self.layer1 = layers.Dense(300, activation='relu')
        self.layer2 = layers.Dense(200, activation='relu')
        self.layer_bn = layers.BatchNormalization()
        self.optimizer = infer_optimizer(self, mode='tensorflow')
        self.criterion = infer_criterion(self.criterion, mode='tensorflow')

    def call(self, x):
        """
        Forward pass through the neural network.
        Args:
            x: Inputs (features) of the neural network.

        Returns:
            Outputs of the neural network pass.
        """
        x = layers.Flatten()(x)
        # x = self.layer1(x)
        # x = self.layer_bn(x)
        # x = self.layer2(x)
        x = self.output_projection(x)
        return x

    def train(self, dataset_dict):
        """
        Trains.
        """
        super()._train(self, dataset_dict)

    def predict(self, dataset_dict) -> dict:
        """
        Predicts.
        """
        return super()._predict(self, dataset_dict)
