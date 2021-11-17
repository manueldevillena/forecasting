import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM1(nn.Module):
    """
    Basic LSTM (RNN) for day-ahead timeseries forecasting.
    """
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        """
        Constructor.
        Args:
            num_classes:
            input_size:
            hidden_size:
            num_layers:
            seq_length:
        """
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


if __name__ == "__main__":

    SHIFT = 24
    PERCENTAGE = 0.7

    num_epochs = 1000
    learning_rate = 0.001
    input_size = 24
    hidden_size = 12
    num_layers = 1
    num_classes = 1

    df = pd.read_csv('data/da_2017_2020.csv', header=0, index_col=0, parse_dates=True, infer_datetime_format=True,
                     dtype=float)

    plt.style.use('ggplot')
    df['DAMprice'].plot(label='CLOSE', title='Belgian Day-Ahead Market Price')
    plt.show()

    X_columns = pd.DataFrame()
    for t in range(SHIFT):
        X_columns[t] = df['DAMprice'].shift(periods=-t)
    y_column = df['DAMprice'].shift(periods=-t - 1)
    X = X_columns.values[:-t - 1]
    y = y_column.values[:-t - 1].reshape(-1, 1)

    ss = StandardScaler()
    mm = MinMaxScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    indices = list(range(len(X_ss)))
    indices_train = indices[:int(PERCENTAGE * len(indices))]
    indices_test = indices[int((PERCENTAGE) * len(indices)):]
    X_train = X_ss[indices_train]
    y_train = y_mm[indices_train]
    X_test = X_ss[indices_test]
    y_test = y_mm[indices_test]

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape)

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
    print(lstm1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm1.forward(X_train_tensors_final)  # forward pass
        optimizer.zero_grad()  # calculates the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e., backpropagation
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    df_X_ss = ss.transform(X)  # old transformers
    df_y_mm = mm.transform(y)  # old transformers

    df_X_ss = Variable(torch.Tensor(df_X_ss))  # converting to Tensors
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))  # reshaping the dataset

    train_predict = lstm1(df_X_ss)  # forward pass
    data_predict = train_predict.data.numpy()  # numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict)  # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    plt.figure(figsize=(10, 6))  # plotting
    # plt.axvline(x=int(PERCENTAGE * len(indices)), c='r', linestyle='--')  # size of the training set

    plt.plot(dataY_plot[int(PERCENTAGE * len(indices)):], label='Actual Data')  # actual plot
    plt.plot(data_predict[int(PERCENTAGE * len(indices)):], label='Predicted Data')  # predicted plot
    plt.title('Day-Ahead Market Prices Prediction')
    plt.legend()
    plt.show()

    MAE = np.mean(np.abs(dataY_plot - data_predict))
    MAE2 = mean_absolute_error(dataY_plot, data_predict)
    MApE = mean_absolute_percentage_error(dataY_plot, data_predict)
