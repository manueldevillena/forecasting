import os
import unittest

from forecast.core import FeatureCreation, Plotter
from forecast.models import SKLRandomForest
from forecast.utils import read_config
import pandas as pd

class TestSKlearn(unittest.TestCase):
    """
    Tests.
    """
    def setUp(self):
        # Set the working directory to the root
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.inputs = 'data/BE.csv'
        self.model_config_path = 'configurations/config_sklearn.yml'
        self.data_config_path = 'configurations/config_data.yml'
        self.data_config = read_config(self.data_config_path)
        self.output = 'tests/results/sklearn'
        os.makedirs(self.output, exist_ok=True)

    def test_random_forests(self):
        """
        Runs random forest
        """
        train_start_date = self.data_config['train_start_date']
        train_end_date = self.data_config['train_end_date']
        val_start_date = self.data_config['val_start_date']
        val_end_date = self.data_config['val_end_date']
        test_start_date = self.data_config['test_start_date']
        test_end_date = self.data_config['test_end_date']

        input_data = FeatureCreation(path_inputs=self.inputs, path_config=self.data_config_path,
                                     train_start_date=train_start_date, train_end_date=train_end_date,
                                     val_start_date=val_start_date, val_end_date=val_end_date,
                                     test_start_date=test_start_date, test_end_date=test_end_date)

        plot = Plotter(self.output, plot_start_date=pd.to_datetime(test_start_date),
                       plot_end_date=pd.to_datetime(test_end_date), fig_name='Day-ahead electricity prices')

        model_hyperparameters = read_config(self.model_config_path)
        model = SKLRandomForest(model_hyperparameters)

        input_data.create_dataset(mode='train')
        model.train(input_data.features)

        tmp_train_start_date = pd.to_datetime(train_start_date)
        tmp_test_start_date = pd.to_datetime(test_start_date)
        tmp_test_end_date = tmp_test_start_date + pd.Timedelta(hours=23)

        today = tmp_test_start_date - pd.Timedelta(hours=13)

        MAX_ITER = 1000
        predictions = {}
        for i in range(MAX_ITER):
            print(i)
            if tmp_test_end_date > pd.to_datetime(test_end_date):
                break
            # retrain on 20th of each month
            if today.day == 20:
                tmp_val_end_date = today - pd.Timedelta(hours=1)
                tmp_val_start_date = tmp_val_end_date - pd.Timedelta(hours=24 * 365)

                tmp_train_start_date = tmp_train_start_date
                tmp_train_end_date = tmp_val_start_date - pd.Timedelta(hours=1)

                input_data.create_dataset(mode='train', train_start_date=tmp_train_start_date,
                                          train_end_date=tmp_train_end_date,
                                          val_start_date=tmp_val_start_date, val_end_date=tmp_val_end_date)
                model.train(input_data.features)

            st = today - pd.Timedelta(hours=168)
            input_data.create_dataset(test_start_date=st, test_end_date=tmp_test_end_date, mode='eval')
            predictions['y_hat'] = model.predict(input_data.features)
            predictions['y'] = input_data.features['y_test']
            predictions['end'] = tmp_test_end_date
            predictions['start'] = tmp_test_start_date
            plot.store_results(predictions)

            today = today + pd.Timedelta(hours=24)
            tmp_test_start_date += pd.Timedelta(hours=24)
            tmp_test_end_date += pd.Timedelta(hours=24)

        plot.plot_predictions('test_day_ahead_BE', zoom=True)
