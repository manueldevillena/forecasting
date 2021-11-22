import os
import unittest

from forecast.core import FeatureCreation, Metrics, Plotter
from forecast.models import TorchLSTM


class TestTorchLSTM(unittest.TestCase):
    """
    Tests.
    """
    def setUp(self):
        # Set the working directory to the root
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.inputs = 'data/da_2017_2020.csv'
        self.config = 'configurations/config.yml'
        self.output = 'tests/results'
        os.makedirs(self.output, exist_ok=True)

    def test_lstm(self):
        """
        Runs torch lstm.
        """
        features = FeatureCreation(path_inputs=self.inputs, path_config=self.config)
        model = TorchLSTM(features)
        model.train()
        predictions = model.predict()
        plot = Plotter(predictions, self.output, features)
        plot.plot_predictions('test_day_ahead', zoom=True)
