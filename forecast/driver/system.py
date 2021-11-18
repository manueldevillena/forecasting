import torch

from abc import ABC, abstractmethod
from torch.autograd import Variable


class System(ABC):
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

    @staticmethod
    def create_tensors(X_train, y_train): #, X_test, y_test):
        """
        Creates torch tensors for torch models.

        Args:
            X_train: Array with inputs to train
            y_train: Array with targets to train
            X_test: Array with inputs to test
            y_test: Array with target to test
        Returns:
            Torch tensors for X_train, y_train, X_test, and y_test
        """
        X_train_tensors = Variable(torch.Tensor(X_train))
        # X_test_tensors = Variable(torch.Tensor(X_test))
        y_train_tensors = Variable(torch.Tensor(y_train))
        # y_test_tensors = Variable(torch.Tensor(y_test))

        X_train_tensors_reshaped = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
        # X_test_tensors_reshaped = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        return X_train_tensors_reshaped, y_train_tensors  #, X_test_tensors_reshaped, y_test_tensors
