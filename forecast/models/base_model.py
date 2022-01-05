from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Interface to system driver.
    """
    def __init__(self):
        """
        Constructor.
        """
        pass

    @abstractmethod
    def train(self, dataset_dict):
        """
        Trains the neural network.

        Args:
            dataset_dict: Dictionary containing keys 'X_train_scaled' and 'y_train_scaled'
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset_dict):
        """
        dataset_dict: Dictionary containing the key 'X_test_scaled'
        """
        raise NotImplementedError
