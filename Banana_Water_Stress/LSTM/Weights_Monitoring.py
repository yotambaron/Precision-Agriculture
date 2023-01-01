import numpy as np
import pandas as pd
import random
from numpy.random.mtrand import RandomState
from tensorflow.keras import *
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)


# ------------------------------------------ Create Synthetic Data --------------------------------------------------- #

# Functions
def cf1(df):
    return df[0] < 0.8 and df[1] < 0.8 and df[2] < 0.8
    # return df[0] * df[1] + df[2] < 0.75
    # return df[0] * df[1] + df[2] * df[3] + 2 * df[4] - df[5] <= 1.75


def cf2(df):
    return df[3] > 0.2 and df[4] > 0.2 and df[5] > 0.2
    # return df[3] * df[4] + df[5] < 0.75
    # return df[6] * df[1] + df[2] * df[3] + 2 * df[4] <= 2.25


def weights_differences(initial, stable, drift):
    weights_summary = pd.DataFrame(columns=['Features', 'Stable Weights Difference', 'Drift Weights Difference'])
    features_names = []
    for j in range(1, len(initial) + 1):
        features_names.append("X" + str(j))
    stable_difference = np.abs(initial - stable).sum(axis=1)
    drift_difference = np.abs(initial - drift).sum(axis=1)
    weights_summary['Features'] = features_names
    weights_summary['Stable Weights Difference'] = stable_difference
    weights_summary['Drift Weights Difference'] = drift_difference
    weights_summary['Drift/Stable Ratio'] = weights_summary['Drift Weights Difference'] / weights_summary['Stable Weights Difference']
    return weights_summary


# Set Parameters
drift_size = 1000
samples = 15000
change_point = 10000
features = 10

# Set random state and create dataframe
rand = RandomState(seed=2)
data = pd.DataFrame(rand.uniform(0, 1, size=(samples, features)))
data.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]
y = np.zeros([samples, 1], dtype=int)

# Create labels according to the 2 cfs and the position in the data stream
for i in range(samples):

    # Before the change point classify by cf1
    if i < change_point:
        if cf1(data.iloc[i]):
            y[i] = 1

    # After the change point classify by cf1 or cf2 depending on the current point and the drift's size
    else:
        p = 1 / (1 + np.exp(-4 * (i - change_point) / drift_size))  # Probability to classify by cf2
        if p < random.uniform(0, 1):  # new concept
            if cf2(data.iloc[i]):
                y[i] = 1
        else:   # Old concept
            if cf1(data.iloc[i]):
                y[i] = 1

data['Target'] = y

print(data.describe())


# ------------------------------------------------ Build Model ------------------------------------------------------- #

# define and fit the base model
def get_base_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(5, input_dim=10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=300, verbose=1)
    return model


y = data.loc[:, "Target"]
train_y = y[0:7000]
del data['Target']
train_x = data.loc[0:6999, :]
nn_model = get_base_model(train_x, train_y)
weights_initial = nn_model.get_weights()[0]

train_y = y[7000:10000]
train_x = data.loc[7000:9999]
nn_model.fit(train_x, train_y, epochs=100, verbose=1)
weights_stable = nn_model.get_weights()[0]

train_y = y[11000:]
train_x = data.loc[11000:]
nn_model.fit(train_x, train_y, epochs=100, verbose=1)
weights_drift = nn_model.get_weights()[0]

weights_difference = weights_differences(weights_initial, weights_stable, weights_drift)
print(weights_difference)
