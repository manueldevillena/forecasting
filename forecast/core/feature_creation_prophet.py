import numpy as np
import pandas as pd

from forecast.core import ForecastInputData
from forecast.utils import read_config

class InvalidInputsError(Exception):
    pass


class FeatureCreationProphet(ForecastInputData):
    """
    Creates the features.
    """

    def __init__(self, path_inputs: str, path_config: str, train_start_date: str, train_end_date: str,
                                   test_start_date: str, test_end_date: str, freq: str = 'H'):
        """
        Constructor.
        """
        super().__init__(path_inputs, freq)
        self.config = read_config(path_config)
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        self.train_df = self.raw_inputs[self.train_start_date:self.train_end_date]
        self.test_df = self.raw_inputs[self.test_start_date:self.test_end_date]

        self.target_columns = self.config['target_columns']

        if len(self.target_columns) > 1:
            raise InvalidInputsError("Only one target variable supported")

        if self.config['covariate_columns'] is None:
            self.covariate_columns = []
        else:
            self.covariate_columns = self.config['covariate_columns']

        self.features = {
                         'train_df': None,
                         'test_df': None,
                          'y_test': None
        }

    def create_dataset(self, train_start_date: str = None, train_end_date: str = None,
                       test_start_date: str = None, test_end_date: str = None,
                       mode: str = 'eval'):

        if train_start_date is not None and train_end_date is not None:
            self.train_start_date = train_start_date
            self.train_end_date = train_end_date

        if test_start_date is not None and test_end_date is not None:
            self.test_start_date = test_start_date
            self.test_end_date = test_end_date

        self.train_df = self.raw_inputs[self.train_start_date:self.train_end_date]
        self.test_df = self.raw_inputs[self.test_start_date:self.test_end_date]


        if mode == 'train':
            self.features = {
                             'train_df': None,
                             'test_df': None,
                              'y_test': None
            }
            df = pd.DataFrame(index=[x for x in range(len(self.train_df))], data=np.nan, columns=['ds', 'y'] + self.covariate_columns)
            df['ds'] = self.train_df.index
            for x in self.target_columns:
                df['y'] = self.train_df[x].values
            for x in self.covariate_columns:
                df[x] = self.train_df[x].values
            self.features['train_df'] = df

        else:
            df = pd.DataFrame(index=[x for x in range(len(self.test_df))], data=np.nan,
                              columns=['ds'] + self.covariate_columns)
            df['ds'] = self.test_df.index
            for x in self.covariate_columns:
                df[x] = self.test_df[x].values
            self.features['test_df'] = df
            for x in self.target_columns:
                self.features['y_test'] = self.test_df[x].values



