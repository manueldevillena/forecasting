import matplotlib.pyplot as plt
import os
from forecast.core import FeatureCreation, Metrics

plt.style.use('ggplot')


class Plotter(Metrics):
    """
    Collection of methods to plot.
    """
    def __init__(self, data_to_plot: dict, output_path: str, features: FeatureCreation):
        """
        Constructor.
        """
        super().__init__(data_to_plot, features)
        self.data_to_plot = data_to_plot
        self.output_path = output_path
        self.features = features

    def plot_predictions(self, name: str = 'default', zoom: bool = False):
        """
        Plots predictions.
        """
        actual_values = self.data_to_plot['actual_values_numpy']
        predicted_values = self.data_to_plot['predicted_values_numpy']
        length_data_set = len(actual_values)
        MAE = self.metrics['MAE']
        # MAPE = self.metrics['MAPE']
        MSE = self.metrics['MSE']

        fig = plt.figure(figsize=(10, 6))
        plt.axvline(x=int(self.percentage * length_data_set), c='r', linestyle='--')
        plt.plot(actual_values, label='Actual Data')
        plt.plot(predicted_values, label='Predicted Data')
        plt.title('Day-Ahead Market Prices Prediction')
        plt.annotate('MAE: {:.2f}'.format(MAE), (int(length_data_set*0.8), int(max(actual_values)*0.7)))
        # plt.annotate('MAPE: {:.2f}'.format(MAPE), (int(length_data_set*0.8), int(max(actual_values)*0.6)))
        plt.annotate('MSE: {:.2f}'.format(MSE), (int(length_data_set*0.8), int(max(actual_values)*0.6)))
        plt.legend()
        fig.savefig(os.path.join(self.output_path, '{}.pdf'.format(name)))

        if zoom:
            zoom = (int(self.percentage * length_data_set), int(self.percentage * length_data_set) + 120)
            self._plot_predictions_zoom(zoom, name)

    def _plot_predictions_zoom(self, zoom: tuple, name: str):
        """
        Plots predictions.
        """
        actual_values = self.data_to_plot['actual_values_numpy']
        predicted_values = self.data_to_plot['predicted_values_numpy']

        fig = plt.figure(figsize=(10, 6))
        plt.plot(actual_values[zoom[0]:zoom[1]], label='Actual Data')
        plt.plot(predicted_values[zoom[0]:zoom[1]], label='Predicted Data')
        plt.title('Day-Ahead Market Prices Prediction')
        plt.legend()
        fig.savefig(os.path.join(self.output_path, '{}_zoom.pdf'.format(name)))
