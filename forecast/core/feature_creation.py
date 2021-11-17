import pandas as pd

from forecast.core import ForecastInputData
from forecast.utils import infer_scaler

class InvalidInputsError(Exception):
    pass


class FeatureCreation(ForecastInputData):
    """
    Creates the features.
    """
    def __init__(self, path_inputs: str, path_config: str, targets: pd.DataFrame = None):
        """
        Constructor.
        """
        super().__init__(path_inputs, path_config)
        self.shift = self.config['shift']
        self.percentage_train = self.config['pertentage_train']
        self.scaler_inputs = infer_scaler(self.config['scaler_inputs'])
        self.scaler_targets = infer_scaler(self.config['scaler_targets'])

        if targets is not None:
            self.X_raw = self.data.drop(targets)
            self.y_raw = self.data[targets]
        if not targets and self.shift > 0:
            self.X_raw = self.data
            self.y_raw = self.data
        else:
            raise InvalidInputsError("There should be independent targets or a shift to create autoregressive models.")

    def _create_features(self):
        """
        Creates the features to be used in the training and testing.
        """
        X, y = self._define_inputs_targets()
        X_scaled, y_scaled = self._scale_data(X, y)
        X_train, y_train, X_test, y_test = self._split_train_test(X_scaled, y_scaled)

    def _define_inputs_targets(self):
        """
        Creates features.
        """
        if self.shift > 0:
            X_columns = pd.DataFrame()
            for t in range(self.shift):
                X_columns[t] = self.X_raw.shift(periods=-t)
            y_column = self.y_raw.shift(periods=-t-1)
            X = X_columns.values[:-t-1]
            y = y_column.values[:-t-1]
        else:
            # TODO: add features in addition to shifting
            X = self.X_raw.values
            y = self.y_raw.values

        return X, y

    def _scale_data(self, X, y):
        """
        Scales the features.
        """
        X_scaled = self.scaler_inputs.fit_transform(X)
        y_scaled = self.scaler_targets.fit_transform(y)

    def _split_train_test(self, X, y):
        """
        Splits the inputs and targets into train and test.
        Args:
            X: Inputs
            y: Targets
            percentage_train: Percentage of train set

        Returns:
            Splits X_train, y_train, X_test, y_test
        """
        indices = list(range(len(X)))
        indices_train = indices[:int(self.percentage_train * len(indices))]
        indices_test = indices[int((self.percentage_train) * len(indices)):]
        X_train = X[indices_train]
        y_train = y[indices_train]
        X_test = X[indices_test]
        y_test = y[indices_test]

        return X_train, y_train, X_test, y_test


