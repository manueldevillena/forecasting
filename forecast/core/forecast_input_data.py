import pandas as pd

from forecast.utils import read_inputs, read_config


class ForecastInputData:
    """
    Loads the input data of the simulation.
    """
    def __init__(self, path_inputs: str, path_config: str,
                 start_date: str = None, end_date: str = None, freq: str = 'H'):
        """
        Constructor.

        Args:
            path_inputs: Path to input csv file
            path_config: Path to input configuration yml file
            start_date: Start date of the simulation
            end_date: End date of the simulation
            freq: Frequancy desired for the simulation
        """
        self.raw_inputs = read_inputs(path_inputs)
        self.start_date = start_date if start_date else self.raw_inputs.index[0]
        self.end_date = end_date if end_date else self.raw_inputs.index[-1]
        self.freq = freq
        self.data = self.raw_inputs  # self._clean_data()
        self.config = read_config(path_config)

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
