from forecast.core import FeatureCreation
from forecast.utils import compute_metrics


class Metrics:
    """
    Collection of methods computing different metrics.
    """
    def __init__(self, data, features: FeatureCreation):
        """
        Constructor.
        """
        self.data = data
        self.features = features
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict:
        """
        Computes metrics.

        Returns:
            Dictionary with metrics
        """
        percentage_train = self.features.percentage_train
        percentage_val = self.features.percentage_validation
        percentage_test = 1 - percentage_train - percentage_val

        actual_values_all = self.data['actual_values_numpy']
        predicted_values_all = self.data['predicted_values_numpy']
        length_data = len(actual_values_all)
        train_actual_values = actual_values_all[:int(percentage_train * length_data)]
        train_predicted_values = predicted_values_all[:int(percentage_train * length_data)]
        val_actual_values = actual_values_all[int(percentage_train * length_data):int(percentage_train * length_data)+int(percentage_val * length_data)]
        val_predicted_values = predicted_values_all[int(percentage_train * length_data):int(percentage_train * length_data)+int(percentage_val * length_data)]
        test_actual_values = actual_values_all[-int(percentage_test * length_data):]
        test_predicted_values = predicted_values_all[-int(percentage_test * length_data):]

        metrics_train = compute_metrics(train_actual_values, train_predicted_values)
        metrics_val = compute_metrics(val_actual_values, val_predicted_values)
        metrics_test = compute_metrics(test_actual_values, test_predicted_values)

        return {
            'metrics_train': metrics_train,
            'metrics_val': metrics_val,
            'metrics_test': metrics_test
        }
