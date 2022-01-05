import numpy as np
import pandas as pd

from forecast.core import ForecastInputData
from forecast.utils import infer_scaler, read_config, scale_data
import ast

class InvalidInputsError(Exception):
    pass


class FeatureCreation(ForecastInputData):
    """
    Creates the features.
    """

    def __init__(self, path_inputs: str, path_config: str, train_start_date: str, train_end_date: str,
                                   val_start_date: str, val_end_date: str,
                                   test_start_date: str, test_end_date: str, freq: str = 'H'):
        """
        Constructor.
        """
        super().__init__(path_inputs, freq)
        self.config = read_config(path_config)

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        self.train_df = self.raw_inputs[self.train_start_date:self.train_end_date]
        self.val_df = self.raw_inputs[self.val_start_date:self.val_end_date]
        self.test_df = self.raw_inputs[self.test_start_date:self.test_end_date]

        self.info = self.config['info']
        keys = list(self.info.keys())
        self.df_dict = {}

        self.input_columns = self.config['input_columns']
        self.target_columns = self.config['target_columns']

        self.scaler_inputs = {}
        for x in keys:
            self.scaler_inputs[x] = infer_scaler(self.config['scaler_inputs'])

        self.scaler_targets = infer_scaler(self.config['scaler_targets'])

        if len(self.input_columns) == 0:
            raise InvalidInputsError("Specify atleast one input column")
        if len(self.target_columns) > 1:
            raise InvalidInputsError("Only one target variable supported")

        self.features = {'y_scaler': None,
                         'X_train_scaled': None,
                         'y_train_scaled': None,
                         'X_val_scaled': None,
                         'y_val_scaled': None,
                         'X_test_scaled': None,
                         'y_test_scaled': None,
                         'y_test': None}

    def _create_features(self, df: pd.DataFrame):
        self.df_dict = {}
        keys = list(self.info.keys())
        LEN_DF = len(df)
        for key in keys:
            self.df_dict[key] = pd.DataFrame(data=np.nan, index=[x for x in range(LEN_DF)],
                                        columns=[x for x in range(abs(self.info[key]['shift']))])

        for i in range(LEN_DF):
            condition = True
            for key in keys:
                if i + self.info[key]['lag'] + self.info[key]['shift'] < 0:
                    condition = False
                    break
                if i + self.info[key]['lag'] + self.info[key]['shift'] > LEN_DF:
                    condition = False
                    break
            if not condition:
                continue

            for key in keys:
                v1_ = i + self.info[key]['lag'] + self.info[key]['shift']
                v2_ = i + self.info[key]['lag']
                if v1_ > v2_:
                    v1 = v2_
                    v2 = v1_
                else:
                    v1 = v1_
                    v2 = v2_

                col_name = key.split('_')[0]
                self.df_dict[key].iloc[i, :] = df[col_name].iloc[v1:v2].transpose()

        for key in keys:
            self.df_dict[key].dropna(axis=0, inplace=True)

    def _scale_and_format_features(self, train=False):
        for key in list(self.df_dict.keys()):
            if 'input' in key:
                self.df_dict[key] = scale_data(self.df_dict[key].values, self.scaler_inputs[key], train=train)
            if 'target' in key:
                self.df_dict[key] = scale_data(self.df_dict[key].values, self.scaler_targets, train=train)

        return self._process_features()

    def _process_features(self):
        X = None
        for key in list(self.df_dict.keys()):
            if 'input' in key:
                if X is None:
                    X = self.df_dict[key]
                else:
                    X = np.concatenate((X, self.df_dict[key]), axis=1)
            if 'target' in key:
                y = self.df_dict[key]

        return X, y


    def create_dataset(self, train_start_date: str = None, train_end_date: str = None,
                       val_start_date: str = None, val_end_date: str = None,
                       test_start_date: str = None, test_end_date: str = None,
                       mode: str = 'test'):

        if train_start_date is not None and train_end_date is not None:
            self.train_start_date = train_start_date
            self.train_end_date = train_end_date
        if val_start_date is not None and val_end_date is not None:
            self.val_start_date = val_start_date
            self.val_end_date = val_end_date
        if test_start_date is not None and test_end_date is not None:
            self.test_start_date = test_start_date
            self.test_end_date = test_end_date

        self.train_df = self.raw_inputs[self.train_start_date:self.train_end_date]
        self.val_df = self.raw_inputs[self.val_start_date:self.val_end_date]
        self.test_df = self.raw_inputs[self.test_start_date:self.test_end_date]

        if mode == 'train':
            self.features = { 'y_scaler': None,
                             'X_train_scaled': None,
                             'y_train_scaled': None,
                             'X_val_scaled': None,
                             'y_val_scaled': None,
                             'X_test_scaled': None,
                             'y_test_scaled': None,
                             'y_test': None
            }

            self._create_features(self.train_df)
            self.features['X_train_scaled'], self.features['y_train_scaled'] = \
                self._scale_and_format_features(train=True)
            self.features['y_scaler'] = self.scaler_targets

            self._create_features(self.val_df)
            self.features['X_val_scaled'], self.features['y_val_scaled'] = \
                self._scale_and_format_features(train=False)
        else:
            self._create_features(self.test_df)
            self.features['y_test'] = self.df_dict[self.target_columns[0]+'_target'].values
            self.features['X_test_scaled'], self.features['y_test_scaled'] = \
                self._scale_and_format_features(train=False)

