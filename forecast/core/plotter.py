import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
from forecast.utils import compute_metrics
from matplotlib.patches import Rectangle
plt.style.use('ggplot')


class Plotter():
    """
    Collection of methods to plot.
    """
    def __init__(self, output_path: str, plot_start_date: str, plot_end_date: str, fig_name: str):
        """
        Constructor.
        """
        self.output_path = output_path
        self.start_date = plot_start_date
        self.end_date = plot_end_date
        self.result = pd.DataFrame(data=np.nan, index=pd.date_range(start=plot_start_date, end=plot_end_date, freq='H'),
                                   columns=['y', 'y_hat'])
        self.fig_name = fig_name

    def store_results(self, dct):
        self.result.loc[dct['start']:dct['end'], 'y'] = dct['y']
        self.result.loc[dct['start']:dct['end'], 'y_hat'] = dct['y_hat']

    def plot_predictions(self, name: str = 'default', zoom: bool = False):
        """
        Plots predictions.
        """
        self.result.dropna(axis=0, inplace=True)
        self.result = self.result.round(3)
        metrics_test = compute_metrics(self.result['y'].values, self.result['y_hat'].values)
        self.result.to_csv(os.path.join(self.output_path, f'{name}.csv'))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        # trans = ax.get_xaxis_transform()
        ax.set_title(self.fig_name)
        # train_limit = ax.axvline(x=int(percentage_train * length_data_set), c='r', linestyle='--')
        # val_limit = ax.axvline(x=int(percentage_val * length_data_set), c='g', linestyle='--')
        # self.result.plot(ax=ax)
        actual, = ax.plot(self.result['y'])
        predicted, = ax.plot(self.result['y_hat'])
        # aux, = ax.plot([0, 0], alpha=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        metrics = [
            # f"MAE_train = {metrics_train['MAE']:.2f}; MSE_train = {metrics_train['MSE']:.2f}",
            # f"MAE_val = {metrics_val['MAE']:.2f}; MSE_val = {metrics_val['MSE']:.2f}",
            f"MAE = {metrics_test['MAE']:.2f}; MSE = {metrics_test['MSE']:.2f}"
        ]
        # fig.text(0, 0.9, metrics[0], bbox=dict(facecolor='red', alpha=0.5), transform=ax.transAxes)
        legend = ax.legend(
                [actual, predicted, extra],
                [
                    # 'Training set',
                    # 'Validation set',
                    'Actual data',
                    'Predicted data',
                    f"{os.linesep.join(map(str, metrics))}"
                ],
                bbox_to_anchor=(1, -0.05)
            )
        # legend = ax.legend(
        #     [train_limit, val_limit, actual, predicted, aux],
        #     [
        #         'Training set',
        #         'Validation set',
        #         'Actual data',
        #         'Predicted data',
        #         f"{os.linesep.join(map(str, metrics))}"
        #     ],
        #     bbox_to_anchor=(1, -0.05)
        # )
        items = [i for i in legend.legendHandles]
        items[-1].set_visible(False)
        fig.savefig(os.path.join(self.output_path, f'{name}.pdf'))

        # if zoom:
        #     zoom = (int(percentage_train * length_data_set), int(percentage_train * length_data_set) + 120)
        #     self._plot_predictions_zoom(zoom, name)

    # def _plot_predictions_zoom(self, zoom: tuple, name: str):
    #     """
    #     Plots predictions.
    #     """
    #     fig = plt.figure(figsize=(10, 6))
    #     plt.plot(self.actual_values[zoom[0]:zoom[1]], label='Actual Data')
    #     plt.plot(self.predicted_values[zoom[0]:zoom[1]], label='Predicted Data')
    #     plt.title('Day-Ahead Market Prices Prediction')
    #     plt.legend()
    #     fig.savefig(os.path.join(self.output_path, f'{name}_zoom.pdf'))
