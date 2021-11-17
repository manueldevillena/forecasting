import argparse
import matplotlib.pyplot as plt
from forecast.core import ForecastInputData, FeatureCreation
from forecast.models import FeedForwardRegression

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parsing the inputs to run the module.")
    parser.add_argument('-d', '--input_data', dest="input_data", help="Input data file (csv).")

    args = parser.parse_args()

    inputs = ForecastInputData(path_inputs=args.input_data)
    features = FeatureCreation(inputs.inputs, targets=None, shift=24)

    network = FeedForwardRegression(features.x, features.y, layers=(512, 512, 512), batch_size=48, num_epochs=50)
    network.fit()
    y = network.predict(features.x)

    plt.plot(y)
    plt.plot(features.y)
    plt.show()