import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if np.isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

dataframe = pd.read_csv(sys.argv[1], delimiter=';', infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# mark all missing values
dataframe.replace('?', np.nan , inplace=True)
# make dataset numeric
dataframe = dataframe.astype('float32')
# fill missing
fill_missing(dataframe.values)

df = dataframe['Global_active_power']

steps = 60
X, y = split_sequence(df.to_numpy(), n_steps=steps)
#print("Number of input-output samples :",np.shape(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 42)

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import variable

##### Model with two hidden layers #####
model_2layer = Sequential()
model_2layer.add(Dense(100, activation='relu', input_dim=steps))
model_2layer.add(Dense(100, activation='relu', input_dim=100))
model_2layer.add(Dense(1))
model_2layer.compile(optimizer='adam', loss='mse')

model_2layer.fit(X_train, y_train, epochs=5, batch_size = 64, verbose=0)

y_pred_2layer = model_2layer.predict(X_test, verbose=0)
print(y_pred_2layer, end = '\n')

'''
fsigmoidrom sklearn.metrics import mean_squared_error, r2_score

y_test = y_test.reshape(y_test.shape[0],1)
print("Mean squared error with two layers: %.2f" % mean_squared_error(y_test, y_pred_2layer))
print('Variance score with two layers: %.2f' % r2_score(y_test, y_pred_2layer))
'''