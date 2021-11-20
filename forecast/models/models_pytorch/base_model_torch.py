import logging
import time
import torch

from torch.autograd import Variable

from forecast.models import BaseModel


class BaseModelTorch(BaseModel):
    """
    Collection of methods used by all torch models.
    """
    def __init__(self):
        """
        Constructor.
        """
        pass

    def train(self):
        tic = time.time()
        running_loss = 0.0
        for epoch in range(self.num_epochs):  # Iterate over number of epochs

            outputs = self.forward(self.X_tensor)  # forward pass
            self.optimizer.zero_grad()  # calculates the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs, self.y_tensor)

            loss.backward()  # calculates the loss of the loss function

            self.optimizer.step()  # improve from loss, i.e., backpropagation

            running_loss += loss.item() * self.X_tensor.size(0)
            epoch_loss = running_loss / len(self.y_tensor)

            if epoch % 100 == 0:
                logging.info('-' * 20)
                logging.info('Epoch: {}/{}'.format(epoch, self.num_epochs - 1))
                logging.info("Loss: {:.4f}".format(epoch_loss))
                logging.info('-' * 20)
                # print("Epoch: {}, loss: {:4f}".format(epoch, loss.item()))
            elif epoch == self.num_epochs - 1:
                logging.info('-' * 20)
                logging.info('-' * 20)
                logging.info("Final loss: {:4f}".format(loss.item()))

        tac = time.time() - tic
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(tac // 60, tac % 60))

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
