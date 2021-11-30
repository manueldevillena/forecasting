from forecast.core import FeatureCreation
from forecast.models.models_tensorflow import BaseModelTF
from forecast.utils import infer_optimizer, infer_criterion
import tensorflow as tf
import keras.layers as layers


class TFFeedForward(tf.keras.Model, BaseModelTF):
    """
    Feed forward network for day-ahead timeseries forecasting.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        Args:
            features: Object with appropriate configuration files.
        """
        tf.keras.Model.__init__(self)
        BaseModelTF.__init__(self, features)

        self.output_projection = layers.Dense(self.size_output)
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
        x = self.output_projection(x)
        return x

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            output = self(x, training=True)
            loss_value = self.criterion(y, output)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value

    def train(self):
        """
        Trains.
        """
        super()._train(self)

    def predict(self) -> dict:
        """
        Predicts.
        """
        return super()._predict(self)
