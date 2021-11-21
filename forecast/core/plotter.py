import matplotlib.pyplot as plt
import os
from forecast.core import FeatureCreation


class Plotter:
    """
    Collection of methods to plot.
    """
    def __init__(self, data_to_plot: dict, output_path: str, features: FeatureCreation):
        """
        Constructor.
        """
        self.data_to_plot = data_to_plot
        self.output_path = output_path
        self.features = features
        self.percentage = features.config['percentage']

    def plot_predictions(self, name: str):
        """
        Plots predictions.
        """
        fig = plt.figure(figsize=(10, 6))
        plt.axvline(x=int(self.percentage * len(indices)), c='r', linestyle='--')  # size of the training set
        plt.plot(dataY_plot[int(self.percentage * len(indices)):], label='Actual Data')  # actual plot
        plt.plot(data_predict[int(self.percentage * len(indices)):], label='Predicted Data')  # predicted plot
        plt.title('Day-Ahead Market Prices Prediction')
        plt.legend()
        fig.savefig(os.path.join(self.output_path), '{}.pdf'.format(name))
