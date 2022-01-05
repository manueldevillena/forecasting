import logging
import numpy as np
import time

from abc import ABC
from forecast.core import FeatureCreation
from forecast.models import BaseModel


class BaseModelSKLearn(BaseModel, ABC):
    """
    Collection of methods used by all sklearn models.
    """
    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    @staticmethod
    def _train(model, dataset_dict):
        """
        Trains the pytorch model.
        Args:
            model: Model to be trained
        """
        tic = time.perf_counter()
        model.fit(dataset_dict['X_train_scaled'], dataset_dict['y_train_scaled'])

        tac = time.perf_counter() - tic
        logging.info(f'Training complete in {tac//60:.0f}m {tac%60:.0f}s')

    def _predict(self, model, dataset_dict):
        """
        Predicts using the trained model.
        Args:
            model: Model used to predict.
        """
        predicted_values = model.predict(dataset_dict['X_test_scaled'])

        return predicted_values
