import pandas as pd

from forecast.utils import read_data


class ForecastInputData:
    """
    Loads the input data of the simulation.
    """

    def __init__(self, path_inputs: str, start_date: str = '01-01-2017', end_date: str = '31-07-2020', freq: str = 'H'):
        """
        Constructor.
        """
        self.raw_inputs = read_data(path_inputs)
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.inputs = self._clean_data()

    def _clean_data(self) -> pd.DataFrame:
        """
        Cleaning data.

        :return: Inputs to be used in the forecasting process
        """
        to_dt = lambda x: pd.to_datetime(x, format="%d-%m-%Y")

        self.raw_inputs.fillna('ffill', inplace=True)
        self.raw_inputs = self.raw_inputs[~self.raw_inputs.index.duplicated()]
        self.raw_inputs = self.raw_inputs.reindex(pd.date_range(self.raw_inputs.index[0], self.raw_inputs.index[-1],
                                                                freq='1h'))
        inputs = self.raw_inputs[to_dt(self.start_date):to_dt(self.end_date)]

        return inputs
