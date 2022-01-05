import os
import unittest

from forecast.core import FeatureCreationProphet, Plotter
from forecast.models import ProphetSimple
from forecast.utils import read_config
import pandas as pd

class TestProphet(unittest.TestCase):
    """
    Tests.
    """
    def setUp(self):
        # Set the working directory to the root
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.inputs = 'data/BE.csv'
        self.model_config_path = 'configurations/config_prophet.yml'
        self.data_config_path = 'configurations/config_data_prophet.yml'
        self.data_config = read_config(self.data_config_path)
        self.output = 'tests/results/prophet'
        os.makedirs(self.output, exist_ok=True)

    def test_prophet(self):
        """
        Runs Prophet model
        """
        train_start_date = self.data_config['train_start_date']
        train_end_date = self.data_config['train_end_date']
        test_start_date = self.data_config['test_start_date']
        test_end_date = self.data_config['test_end_date']

        input_data = FeatureCreationProphet(path_inputs=self.inputs, path_config=self.data_config_path,
                                   train_start_date=train_start_date, train_end_date=train_end_date,
                                   test_start_date=test_start_date, test_end_date=test_end_date)

        plot = Plotter(self.output, plot_start_date=pd.to_datetime(test_start_date),
                       plot_end_date=pd.to_datetime(test_end_date), fig_name='Day-ahead electricity prices')

        model_hyperparameters = read_config(self.model_config_path)
        model = ProphetSimple(model_hyperparameters)

        input_data.create_dataset(mode='train')
        model.train(input_data.features)

        tmp_train_start_date = pd.to_datetime(train_start_date)
        tmp_test_start_date = pd.to_datetime(test_start_date)
        tmp_test_end_date = tmp_test_start_date + pd.Timedelta(hours=23)

        today = tmp_test_start_date

        MAX_ITER = 1000
        predictions = {}
        for i in range(MAX_ITER):
            print(i)
            if tmp_test_end_date > pd.to_datetime(test_end_date):
                break

            # retrain on 20th of each month
            if today.day == 20:
                tmp_train_start_date = tmp_train_start_date
                tmp_train_end_date = today - pd.Timedelta(hours=1)

                input_data.create_dataset(mode='train', train_start_date=tmp_train_start_date,
                                          train_end_date=tmp_train_end_date)
                model = ProphetSimple(model_hyperparameters)
                model.train(input_data.features)

            input_data.create_dataset(test_start_date=tmp_test_start_date, test_end_date=tmp_test_end_date, mode='eval')
            predictions['y_hat'] = model.predict(input_data.features).loc[:, 'yhat'].values
            predictions['y'] = input_data.features['y_test']
            predictions['end'] = tmp_test_end_date
            predictions['start'] = tmp_test_start_date
            plot.store_results(predictions)

            today = today + pd.Timedelta(hours=24)
            tmp_test_start_date += pd.Timedelta(hours=24)
            tmp_test_end_date += pd.Timedelta(hours=24)

        plot.plot_predictions('test_day_ahead_BE', zoom=True)