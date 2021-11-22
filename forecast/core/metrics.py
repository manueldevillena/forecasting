from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from forecast.core import FeatureCreation


class Metrics:
    """
    Collection of methods computing different metrics.
    """
    def __init__(self, data, features: FeatureCreation):
        """
        Constructor.
        """
        self.data = data
        self.percentage = features.config['pertentage_train']
        self.metrics = self._compute_metrics()

    def _compute_metrics(self):
        """

        Returns:

        """
        actual_values_all = self.data['actual_values_numpy']
        predicted_values_all = self.data['predicted_values_numpy']
        length_data = len(actual_values_all)
        actual_values = actual_values_all[int(self.percentage * length_data):]
        predicted_values = predicted_values_all[int(self.percentage * length_data):]

        MAE = mean_absolute_error(actual_values, predicted_values)
        MAPE = mean_absolute_percentage_error(actual_values, predicted_values)

        return {
            'MAE': MAE,
            'MAPE': MAPE
        }
