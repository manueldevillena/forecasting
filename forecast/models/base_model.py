import numpy as np

from abc import ABC, abstractmethod
from forecast.utils import numpy_to_torch


class BaseModel(ABC):
    """
    Generic model establishing the interface.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, input_scaler: str, target_scaler: str, percentage_train: float):
        """
        Constructor.

        :param x: Input array
        :param y: Target array
        :param input_scaler: String that specifies what scaler will be used for inputs (e.g., "MinMaxScaler(feature_range(-1,1))")
        :param target_scaler: String that specifies what scaler will be used for the targets (e.g., "StandardScaler()")
        """
        self.input_scaler = eval(input_scaler) if input_scaler is not None else None
        self.target_scaler = eval(target_scaler) if target_scaler is not None else None
        self.num_targets = y.shape[1]
        self._fit_scalers(x, y)
        self.x = self.scale_inputs(x)
        self.y = self.scale_targets(y)
        self.percentage_train = percentage_train
        self.data_sets = self._create_datasets_dic()

    @abstractmethod
    def fit(self):
        """
        Fit the model to the data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.array):
        """
        Predicts with the model given an input x.

        :param x: Input array.
        """
        raise NotImplementedError

    def scale_inputs(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms input array x into scaled array.

        :param x: Input array
        :return: Scaled input array x
        """
        return self.input_scaler.transform(x) if self.input_scaler is not None else x

    def scale_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Transforms target array y into scaled array.

        :param y: Target array
        :return: Scaled target array y
        """
        return self.target_scaler.transform(y) if self.target_scaler is not None else y

    def inverse_scale_inputs(self, scaled_x: np.ndarray) -> np.ndarray:
        """
        Transforms a scaled input array into the original array x

        :param scaled_x: Scaled input array
        :return: Input array
        """

        return self.input_scaler.inverse_transform(scaled_x) if self.input_scaler is not None else scaled_x

    def inverse_scale_targets(self, scaled_y: np.ndarray) -> np.ndarray:
        """
        Transforms a scaled target array into the target array y

        :param scaled_y: Scaled target array
        :return: Target array
        """
        shape = np.shape(scaled_y)
        if len(shape) > 2:
            return np.stack([self.target_scaler.inverse_transform(scaled_y[:, :, i]) for i in range(shape[2])],
                            axis=2) if self.target_scaler is not None else scaled_y
        else:
            return self.target_scaler.inverse_transform(
                scaled_y.reshape(-1, self.num_targets)).reshape(-1, self.num_targets) if self.target_scaler is not None \
                else scaled_y.reshape(-1, self.num_targets)

    def transform_datasets(self) -> bool:
        """
        Transforms original data sets from numpy arrays to torch tensors.

        :return: True if exited normally
        """
        for key, val in self.data_sets.items():
            torch_arr = list()
            for arr in val:
                torch_arr.append(numpy_to_torch(arr))
            self.data_sets[key] = torch_arr

        return True

    def _fit_scalers(self, x: np.ndarray, y: np.ndarray) -> bool:
        """
        Fits scalers to the inputs and the targets.

        :param x: Input array
        :param y: Targets array
        :return: True if exited correctly
        """
        if self.input_scaler is not None:
            self.input_scaler.fit(x)
        if self.target_scaler is not None:
            self.target_scaler.fit(y)
        return True

    def _create_datasets_dic(self) -> dict:
        """
        Splits the dataset into train and validation sets according to the specified percentage_train.

        :return: Dict containing the train set and test set pairs of (inputs, targets)
        """
        x_train, y_train, x_val, y_val = self._train_test_split(self.x, self.y, percentage_train=self.percentage_train)

        return {"train": [x_train, y_train], "val": [x_val, y_val]}

    @staticmethod
    def _train_test_split(X, y, percentage_train):
        """
        Splits the inputs and targets into train and test (or validation).
        Args:
            X: Inputs
            y: Targets
            percentage_train: Percentage of train set

        Returns:
            Splits X_train, y_train, X_test, y_test
        """
        indices = list(range(len(X)))
        indices_train = indices[:int(percentage_train * len(indices))]
        indices_eval = indices[int((percentage_train) * len(indices)):]
        X_train = X[indices_train]
        y_train = y[indices_train]
        X_test = X[indices_eval]
        y_test = y[indices_eval]

        return X_train, y_train, X_test, y_test
