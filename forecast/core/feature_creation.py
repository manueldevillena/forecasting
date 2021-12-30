import numpy as np
import pandas as pd

from forecast.core import ForecastInputData
from forecast.utils import infer_scaler


class InvalidInputsError(Exception):
    pass


class FeatureCreation(ForecastInputData):
    """
    Creates the features.
    """

    def __init__(self, path_inputs: str, path_config: str):
        """
        Constructor.
        """
        super().__init__(path_inputs, path_config)
        self.shift = self.config['shift']
        self.multi_step_forecast = self.config['multi_step_forecast']
        self.percentage_train = self.config['percentage_train']
        self.percentage_validation = self.config['percentage_validation']

        self.input_columns = self.config['input_columns']
        self.target_columns = self.config['target_columns']

        self.scaler_inputs = {}
        for x in range(len(self.input_columns)):
            self.scaler_inputs[x] = infer_scaler(self.config['scaler_inputs'])

        self.scaler_targets = infer_scaler(self.config['scaler_targets'])

        if len(self.input_columns) == 0:
            raise InvalidInputsError("Specify atleast one input column")
        if len(self.target_columns) > 1:
            raise InvalidInputsError("Only one target variable supported")

        if self.config['input_columns_with_known_future_values'] is None:
            self.input_columns_with_known_future_values = []
        else:
            self.input_columns_with_known_future_values = self.config['input_columns_with_known_future_values']
            if not set(self.input_columns_with_known_future_values).issubset(set(self.input_columns)):
                raise InvalidInputsError("Input columns with known future values must be a subset of input columns")

            self.scaler_inputs_with_known_future_values = {}
            for x in range(len(self.input_columns_with_known_future_values)):
                self.scaler_inputs_with_known_future_values[x] = infer_scaler(self.config['scaler_inputs'])

        self.X_raw = self.data[self.input_columns]
        self.y_raw = self.data[self.target_columns]

        self.features = self._create_features()

    def _create_features(self) -> dict:
        """
        Creates the features to be used in the training, validation and testing.
        """
        X, C, y = self._define_inputs_targets()
        X_train, C_train, y_train, X_val, C_val, y_val \
            = self._split_set(X, C, y, self.percentage_train, self.percentage_validation)

        X_train_scaled = self._scale_x_data(X_train, self.scaler_inputs, mode='train')
        y_train_scaled = self._scale_y_data(y_train, self.scaler_targets, mode='train')

        X_val_scaled = self._scale_x_data(X_val, self.scaler_inputs, mode='val')
        y_val_scaled = self._scale_y_data(y_val, self.scaler_targets, mode='val')

        if C_train is None and C_val is None:
            C_train_scaled = None
            C_val_scaled = None
        else:
            C_train_scaled = self._scale_x_data(C_train, self.scaler_inputs_with_known_future_values, mode='train')
            C_val_scaled = self._scale_x_data(C_val, self.scaler_inputs_with_known_future_values, mode='val')

        X = self._scale_x_data(X, self.scaler_inputs, mode='val')

        # notice, X has the shape (samples, past_length, number_of_input_columns),
        # C has the shape (samples, future_length, number_of_input_columns_with_known_future_values) or None
        # for now let us combine them together to get the shape
        # (samples, past_length*number_of_input_columns + future_length*number_of_input_columns_with_known_future_values)

        X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1]*X_train_scaled.shape[2])
        if C_train_scaled is None:
            pass
        else:
            C_train_scaled = C_train_scaled.reshape(-1, C_train_scaled.shape[1] * C_train_scaled.shape[2])
            X_train_scaled = np.concatenate((X_train_scaled, C_train_scaled), axis=1)

        X_val_scaled = X_val_scaled.reshape(-1, X_val_scaled.shape[1] * X_val_scaled.shape[2])
        if C_val_scaled is None:
            pass
        else:
            C_val_scaled = C_val_scaled.reshape(-1, C_val_scaled.shape[1] * C_val_scaled.shape[2])
            X_val_scaled = np.concatenate((X_val_scaled, C_val_scaled), axis=1)

        X = X.reshape(-1, X.shape[1] * X.shape[2])
        if C is None:
            pass
        else:
            C = C.reshape(-1, C.shape[1] * C.shape[2])
            X = np.concatenate((X, C), axis=1)

        features = {
            'X_train_scaled': X_train_scaled,
            'y_train_scaled': y_train_scaled,
            'X_val_scaled': X_val_scaled,
            'y_val_scaled': y_val_scaled,
            'X': X,
            'y': y,
            'X_scaler': self.scaler_inputs,
            'y_scaler': self.scaler_targets
        }

        return features

    def _define_inputs_targets(self):
        """
        Creates features.
        """
        X_return = None
        C_return = None

        features = pd.DataFrame()
        for t in range(self.shift, self.shift + self.multi_step_forecast):
            features[t] = self.y_raw[self.target_columns].shift(-t)
        features.dropna(axis=0, inplace=True)
        y_return = features.values

        for i in self.input_columns:
            features = pd.DataFrame()
            for t in range(self.shift):
                features[t] = self.X_raw[i].shift(-t)
            features.dropna(axis=0, inplace=True)
            features = np.expand_dims(features.values, axis=2)

            if X_return is None:
                X_return = features
            else:
                X_return = np.concatenate((X_return, features), axis=2)

        X_return = X_return[:y_return.shape[0], :, :]

        for i in self.input_columns_with_known_future_values:
            features = pd.DataFrame()
            for t in range(self.shift, self.shift + self.multi_step_forecast):
                features[t] = self.X_raw[i].shift(-t)
            features.dropna(axis=0, inplace=True)
            features = np.expand_dims(features.values, axis=2)

            if C_return is None:
                C_return = features
            else:
                C_return = np.concatenate((C_return, features), axis=2)

        return X_return, C_return, y_return

    @staticmethod
    def _split_set(X, C, y, percentage_train: float = 0.6, percentage_val: float = 0.2) -> tuple:
        """
        Splits the inputs and targets into train and test.
        Args:
            X: Inputs
            y: Targets
            percentage: percentage of train or validation sets

        Returns:
            Splits X_train, y_train, X_test, y_test
        """
        C_train = None
        C_val = None

        indices = list(range(len(X)))
        p1 = int(percentage_train * len(indices))
        p2 = int((percentage_train+percentage_val) * len(indices))
        indices_train = indices[:p1]
        indices_val = indices[p1:p2]

        X_train = X[indices_train]
        y_train = y[indices_train]
        X_val = X[indices_val]
        y_val = y[indices_val]

        if C is not None:
            C_train = C[indices_train]
            C_val = C[indices_val]

        return X_train, C_train, y_train, X_val, C_val, y_val

    def _scale_x_data(self, X, scaler, mode='val') -> tuple:
        """
        Scales the input features.
        """
        scaled = np.empty(shape=X.shape)
        scaled[:] = np.nan
        for i in range(X.shape[-1]):
            if mode == 'train':
                scaled[:, :, i] = scaler[i].fit_transform(X[:, :, i])
            else:
                scaled[:, :, i] = scaler[i].transform(X[:, :, i])

        return scaled

    def _scale_y_data(self, y, scaler, mode='val') -> tuple:
        """
        Scales the input features.
        """
        if mode == 'train':
            scaled = scaler.fit_transform(y)
        else:
            scaled = scaler.transform(y)

        return scaled