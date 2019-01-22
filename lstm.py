# TODO: use batch normalisation instead of dropout
# TODO: Play with timesteps, epochs, batch size, optimiser, scaler, dropout rate
# TODO: Apply gridSearchCV for parameter testing

""" Part 1: Data Preprocessing"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import h5py
from keras.utils import plot_model

# Importing the dataset
dataset_train = pnd.read_csv('Google_Stock_Train.csv')
training_set = dataset_train.iloc[:, [2]].values

# Scaling the dataset
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
scaled_training_dataset = sc.fit_transform(training_set)
# print(scaled_training_dataset.shape)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
Y_train = []
time_steps = 260
for i in range(time_steps, len(scaled_training_dataset)):
    X_train.append(scaled_training_dataset[i - time_steps:i, 0])
    Y_train.append(scaled_training_dataset[i, 0])

# Change the training dataset into numpy array
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # try without reshaping

"""Part 2: Building the RNN"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Initialising the RNN
regressor = Sequential()
no_of_units = 200
# First LSTM layer and dropout regularization
regressor.add(LSTM(units=no_of_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.3))
# regressor.add(BatchNormalization())

# Second LSTM layer and dropout regularization
regressor.add(LSTM(units=no_of_units, return_sequences=True))
regressor.add(Dropout(0.3))
# regressor.add(BatchNormalization())

# Third LSTM layer and dropout regularization
regressor.add(LSTM(units=no_of_units, return_sequences=True))
regressor.add(Dropout(0.3))
# regressor.add(BatchNormalization())

# Fourth LSTM layer and dropout regularization
regressor.add(LSTM(units=no_of_units))
regressor.add(Dropout(0.3))
# regressor.add(BatchNormalization())
# Output layer
regressor.add(Dense(units=1))
plot_model(regressor, to_file='model.png')
# Compile RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# using callbacks
# es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=20, verbose=1)
# rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
check_point = ModelCheckpoint(filepath='weights.hdf5', monitor='loss', verbose=1, save_best_only=True,
                              save_weights_only=True)
# tb = TensorBoard()

# Fitting the RNN to training dataset
regressor.fit(X_train, Y_train, epochs=100, batch_size=32, callbacks=[check_point])

"""Part 3: Making predictions and visualising the data"""

# Getting the real stock price of January 2017
dataset_test = pnd.read_csv('Google_Stock_Test.csv')
real_stock_price = dataset_test.iloc[:, [2]].values

# Getting the predicted stock price of January 2017
dataset_total = pnd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_steps:].values  # why not use iloc method???
inputs = inputs.reshape(-1, 1)  # why (-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(time_steps, len(inputs)):
    X_test.append(inputs[i - time_steps:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Stock Value')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Value')
plt.xlabel('Time')
plt.ylabel('Stock Value')
plt.title('Stock price comparison')
plt.legend()
plt.show()

# import math
# from sklearn.metrics import mean_squared_error
#
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
#
# relative_error = rmse / max(real_stock_price)
#
# print(relative_error)
