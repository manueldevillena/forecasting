import pandas as pd


class InvalidInputsError(Exception):
    pass


class FeatureCreation:
    """
    Creates the features.
    """
    def __init__(self, inputs: pd.DataFrame, targets: str = None, shift: int = 24):
        """
        Constructor.
        """
        if targets is not None:
            self.x_raw = inputs.drop(targets)
            self.y_raw = inputs[targets]
        if not targets and shift > 0:
            self.x_raw = inputs
            self.y_raw = inputs
        else:
            raise InvalidInputsError("There should be independent targets or a shift to create autoregressive models.")

        self.shift = shift
        self.x, self.y = self._create_features()

    def _create_features(self):
        """
        Creates features.
        """
        if self.shift > 0:
            x_columns = pd.DataFrame()
            for t in range(self.shift):
                x_columns[t] = self.x_raw.shift(periods=-t)
            y_column = self.y_raw.shift(periods=-t-1)
            x = x_columns.values[:-t-1]
            y = y_column.values[:-t-1]
        else:
            # TODO: add features in addition to shifting
            x = self.x_raw.values
            y = self.y_raw.values

        return x, y
