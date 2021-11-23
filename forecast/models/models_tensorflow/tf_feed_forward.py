from forecast.core import FeatureCreation
from forecast.models.models_tensorflow import BaseModelTF
from forecast.utils import infer_optimizer, infer_criterion


class TFFeedForward(BaseModelTF):
    """
    Basic LSTM (RNN) for day-ahead timeseries forecasting.
    """
    def __init__(self, features: FeatureCreation):
        """
        Constructor.
        Args:
            features: Object with appropriate configuration files.
        """
        super().__init__(self, features)

        self.seq_length = self.X_train_tensor.shape[1]

        self.lstm = nn.LSTM(input_size=self.size_input, hidden_size=self.size_hidden, num_layers=self.num_layers_lstm,
                            batch_first=True)
        self.linear_layers = self.create_linear_net()
        self.net = nn.Sequential(*self.linear_layers)

        self.optimizer = infer_optimizer(self)
        self.criterion = infer_criterion(self.criterion)

    def forward(self, x):
        """
        Forward pass through the neural network.
        Args:
            x: Inputs (features) of the neural network.

        Returns:
            Outputs of the neural network pass.
        """
        pass


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
