from abc import ABC, abstractmethod

from forecast.models import MethodsBase


class MethodsTorch(MethodsBase, ABC):
    """
    Collection of methods used by all torch models.
    """
    def __init__(self, num_epochs: int, ):
        """
        Constructor.
        """
        self.num_epochs = num_epochs

