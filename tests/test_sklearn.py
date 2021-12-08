import os
import unittest

from forecast.core import FeatureCreation, Plotter
from forecast.models import SKLRandomForest


class TestSKlearn(unittest.TestCase):
    """
    Tests.
    """
    def setUp(self):
        # Set the working directory to the root
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.inputs = 'data/da_2017_2020.csv'
        self.config = 'configurations/config_sklearn.yml'
        self.output = 'tests/results/sklearn'
        os.makedirs(self.output, exist_ok=True)

    def test_random_forests(self):
        """
        Runs torch lstm.
        """
        features = FeatureCreation(path_inputs=self.inputs, path_config=self.config)
        model = SKLRandomForest(features)
        model.train()
        predictions = model.predict()
        plot = Plotter(predictions, self.output, features)
        plot.plot_predictions('test_day_ahead', zoom=True)
