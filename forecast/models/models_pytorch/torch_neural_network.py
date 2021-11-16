import copy
import logging
import numpy as np
import time
import torch

from forecast.models import BaseModel
from forecast.utils import infer_activation, numpy_to_torch

LOSS = 10e15


class TorchNeuralNetwork(BaseModel):
    """
    Neural network interface based on PyTorch.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, input_scaler: str = None, target_scaler: str = None,
                 percentage_train: float = 0.7, batch_size: int = 24, num_epochs: int = 25, activation: str = 'relu',
                 device: str = 'cpu', lr: float = 0.001, weight_decay: float = 0):
        """
        Constructor.

        :param x: Input array
        :param y: Target array
        :param input_scaler: String that specifies what scaler will be used for inputs (e.g., "MinMaxScaler(feature_range(-1,1))")
        :param target_scaler: String that specifies what scaler will be used for the targets (e.g., "StandardScaler()")
        :param percentage_train: Percentage of the input data used as training set
        :param batch_size: Number of data points used per gradient update
        :param num_epochs: Number of passes through the entire train set
        :param activation: Activation function to be used
        :param device: On which device should the experiment run
        :param lr: Learning rate
        :param weight_decay: Decreasing rate of the weights
        """
        super().__init__(x, y, input_scaler, target_scaler, percentage_train)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.activation = infer_activation(activation)
        self.lr = lr
        self.criterion = None
        self.model = None
        self.weight_decay = weight_decay
        # TODO: remove self parameters for lr and weight decay
        self.optimizer = None# torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self):
        """
        Fit the model to the data.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.transform_datasets()
        self.model = self.train_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts with the model given an input x.

        :param x: Input array.
        :return: Predictions array
        """
        x = self.scale_inputs(x)
        x = numpy_to_torch(x)
        self.model.eval()
        self.model.device = self.device
        self.model.to(self.model.device)
        y = self.model(x).data.numpy()

        return self.inverse_scale_targets(y)

    def train_model(self):
        """
        Model training.

        :return: trained model
        """
        tic = time.time()
        self.model.to(self.device)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = LOSS

        indexes = {x: list(range(len(self.data_sets[x][1]))) for x in ['train', 'val']}

        for epoch in range(self.num_epochs):
            logging.info('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            logging.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                n_batches = int(len(self.data_sets[phase][1]) / self.batch_size)
                # Iterate over data.
                for n in range(n_batches):
                    idx = np.random.choice(indexes[phase], size=self.batch_size)

                    inputs = self.data_sets[phase][0][idx].to(self.device)
                    labels = self.data_sets[phase][1][idx].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / len(self.data_sets[phase][1])

                logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        tac = time.time() - tic
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(tac // 60, tac % 60))
        logging.info('Best val Acc: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        return self.model
