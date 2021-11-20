import argparse
import os

import matplotlib.pyplot as plt

from forecast.core import FeatureCreation
from forecast.driver import TorchDriver
from forecast.models import TorchLSTM
from forecast.utils import set_up_logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parsing the inputs to run the module.")
    parser.add_argument("-d", "--input_data", dest="input_data", help="Input data file (csv).")
    parser.add_argument("-c", "--configuration_file", dest="config_file", help="YML file with configuration options")
    parser.add_argument("-o", "--output_path", dest="output", help="Output path for the results.")

    args = parser.parse_args()

    # Prepare output path
    os.makedirs(args.output, exist_ok=True)

    logger = set_up_logger(args.output)
    logger.info('Started')

    features = FeatureCreation(path_inputs=args.input_data, path_config=args.config_file)
    model = TorchLSTM(features)
    model.train()
    model.predict()
    # driver = TorchDriver(features)
    # driver.train(features.X, features.y)
    # network = FeedForwardRegression(features.X, features.y, layers=(512, 512, 512), batch_size=48, num_epochs=50)
    # network.fit()
    # y = network.predict(features.x)
    #
    # plt.plot(y)
    # plt.plot(features.y)
    # plt.show()
