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

    def plot_predictions(self, name: str = 'default', zoom: bool = False):
        """
        Plots predictions.
        """
        percentage_train = self.features.percentage_train
        actual_values = self.data_to_plot['actual_values_numpy'][:, 0]
        predicted_values = self.data_to_plot['predicted_values_numpy'][:, 0]
        length_data_set = len(actual_values)
        metrics_train = self.metrics['metrics_train']
        metrics_val = self.metrics['metrics_val']
        metrics_test = self.metrics['metrics_test']

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        trans = ax.get_xaxis_transform()
        ax.set_title('Day-Ahead Market Prices Prediction')
        ax.axvline(x=int(percentage_train * length_data_set), c='r', linestyle='--')
        actual, = ax.plot(actual_values)
        predicted, = ax.plot(predicted_values)
        aux, = ax.plot([0, 0], alpha=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        metrics = [
            f"MAE_train = {metrics_train['MAE']:.2f}; MSE_train = {metrics_train['MSE']:.2f}",
            f"MAE_val = {metrics_val['MAE']:.2f}; MSE_val = {metrics_val['MSE']:.2f}",
            f"MAE_test = {metrics_test['MAE']:.2f}; MSE_test = {metrics_test['MSE']:.2f}"
        ]
        legend = ax.legend(
            [actual, predicted, aux],
            [
                'Actual data',
                'Predicted data',
                f"{os.linesep.join(map(str, metrics))}"
            ],
            bbox_to_anchor=(1, -0.05)
        )
        items = [i for i in legend.legendHandles]
        items[-1].set_visible(False)
        fig.savefig(os.path.join(self.output_path, f'{name}.pdf'))

        if zoom:
            zoom = (int(percentage_train * length_data_set), int(percentage_train * length_data_set) + 120)
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
        fig.savefig(os.path.join(self.output_path, f'{name}_zoom.pdf'))
