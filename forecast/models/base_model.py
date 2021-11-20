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
    def train(self, X: object, y: object):
        """
        Trains the neural network.

        Args:
            X (object): Input tensor to train (features)
            y (object): Input tensor to train (targets)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        Predicts on new data.
        """
        raise NotImplementedError
