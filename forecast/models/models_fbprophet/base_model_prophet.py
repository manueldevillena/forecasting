import logging
import numpy as np
import time
import abc

import pandas as pd

from forecast.models import BaseModel
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt


class BaseModelProphet(BaseModel, abc.ABC):
    """
    Collection of methods used by all tensorflow (with keras) models.
    """
    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    @staticmethod
    def _train(model, dataset):
        """
        Trains the tensorflow model.
        Args:
            model: Model to be trained
        """
        tic = time.perf_counter()
        model.fit(dataset['train_df'])
        tac = time.perf_counter() - tic
        logging.info(f'Training complete in {tac//60:.0f}m {tac%60:.0f}s')

    def _predict(self, model, dataset):
        """
        Predicts using the trained model.
        Args:
            model: Model used to predict.
        """
        df = model.predict(dataset['test_df'])
        return df
