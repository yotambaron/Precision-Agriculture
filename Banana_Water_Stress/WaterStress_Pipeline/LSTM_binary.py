from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, mean_squared_error, auc, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from itertools import groupby
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import datetime
import pandas as pd
import logging
import numpy as np
import random
import seaborn as sns
import warnings
from sklearn import metrics
import itertools
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.options.mode.chained_assignment = None
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
random_state = 28
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)


# ------------------------- Functions -------------------------------- #

def Base_LSTM_Model(input_size, input_time, active, opt, learn, moment, n_units=200, num_layers=2):
    # Initiate the model
    model = Sequential()

    for i in range(1, num_layers + 1):
        if i != num_layers:
            # Add an LSTM layer - return_sequences = False as we take only last cell hidden state for classification
            model.add(LSTM(units=n_units, input_shape=(input_time, input_size),
                           activation=active, return_sequences=True))
        # if we add another layer we will use return_sequences = True
        # Add a dense layer with one neuron as an output applied to the last cell hidden state
        else:
            model.add(LSTM(units=n_units, input_shape=(input_time, input_size),
                           activation=active, return_sequences=False))
    # ANN layers
    model.add(Dense(n_units, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(n_units, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(n_units, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    if opt == 'SGD':
        opt = SGD(momentum=moment, learning_rate=learn)
    if opt == 'RMSprop':
        opt = RMSprop()
    if opt == 'Adam':
        opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['MAE'])
    # The resulting model
    print(model.summary())
    return model


def cv_LSTM(stable_flag, y_stable, tempo_data, y_t_v, b_size, epochs_num, seq_size, time_dim, unit, layer, active, opt, learn_rate, mom):
    b_acc_per_fold = pd.DataFrame()
    f1_per_fold = pd.DataFrame()
    auc_per_fold = pd.DataFrame()
    n_folds = 4
    threshold = 0.5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1

    for train, test in skf.split(tempo_data, y_t_v):
        print(y_t_v[test])

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        lstm_model = Base_LSTM_Model(seq_size, time_dim, active, opt, learn_rate, mom, n_units=unit, num_layers=layer)
        if stable_flag:
            lstm_model.fit(tempo_data[train], y_stable[train], batch_size=b_size, epochs=epochs_num, verbose=1)
        else:
            lstm_model.fit(tempo_data[train], y_t_v[train], batch_size=b_size, epochs=epochs_num, verbose=1)

        # Generate generalization metrics
        probabilities = lstm_model.predict(tempo_data[test], verbose=1)
        predictions = np.where(probabilities > threshold, 1, 0)
        predictions = predictions.reshape(-1, 1)

        if stable_flag:
            b_acc = balanced_accuracy_score(y_stable[test], predictions)
            f1 = f1_score(y_stable[test], predictions)
            auc = 0.5
        else:
            b_acc = balanced_accuracy_score(y_t_v[test], predictions)
            f1 = f1_score(y_t_v[test], predictions)
            fpr, tpr, thresholds = metrics.roc_curve(y_t_v[test], probabilities, pos_label=1)
            auc = metrics.auc(fpr, tpr)
        b_acc_per_fold = b_acc_per_fold.append({'Validation balanced accuracy': b_acc}, ignore_index=True)
        f1_per_fold = f1_per_fold.append({'Validation f1': f1}, ignore_index=True)
        auc_per_fold = auc_per_fold.append({'Validation auc': auc}, ignore_index=True)

        print(f'Score for fold {fold_no}: balanced accuracy of {round(b_acc, 2)}%')
        print(f'Score for fold {fold_no}: f1 of {round(f1, 2)}%')
        print(f'Score for fold {fold_no}: auc of {round(auc, 2)}%')

        # Increase fold number
        fold_no = fold_no + 1

    avg_acc_per_fold = b_acc_per_fold.mean()
    avg_f1_per_fold = f1_per_fold.mean()
    avg_auc_per_fold = auc_per_fold.mean()
    print(avg_acc_per_fold)
    print(avg_f1_per_fold)
    print(avg_auc_per_fold)
    return avg_acc_per_fold, avg_f1_per_fold, avg_auc_per_fold


def normalization(data):  # normalization by min- max
    norm_data = (data - data.min()) / (data.max() - data.min())
    return norm_data


# --------------------------------------------- Train-Test Analysis -------------------------------------------------- #
#
# # ------------ Load Data ------------- #
# path = r'C:\Users\mannh\Banana'
# path_col = r'C:\Users\mannh\Banana'
# save_path = r'C:\Users\mannh\Banana\Results'
# df = pd.read_csv(path + '/All_features_smoothed.csv')
# cols = pd.read_csv(path_col + '/Disc_Entropy.csv')
# del cols['class']
# cols = cols.columns
#
# # ------------ Set params ------------- #
# plants = 48
# change_point = 14
# days = 41 - change_point + 1
#
# # Remove stable days
# df = df[df['Date'] >= change_point].reset_index()
# df['plant'] = df['Treatment'] + df['Num'].astype(str)
#
# # Chose plants for test set
# test_plants = ['A1', 'A4', 'A17', 'A18', 'A21', 'A22', 'A31', 'A36', 'A40', 'A45',
#                'B1', 'B4', 'B17', 'B18', 'B21', 'B22', 'B31', 'B36', 'B40', 'B45',
#                'C1', 'C4', 'C17', 'C18', 'C21', 'C22', 'C31', 'C36', 'C40', 'C45',
#                'D1', 'D4', 'D17', 'D18', 'D21', 'D22', 'D31', 'D36', 'D40', 'D45']
#
# del df['Treatment']
# del df['Num']
# del df['index']
#
# # Split to train and test
# train_data = df[~df['plant'].isin(test_plants)]
# test_data = df[df['plant'].isin(test_plants)]
#
# # Get y for train
# temp = train_data.copy()
# temp = temp.groupby('plant').sum()
# y_train = np.array(temp.index)
# y_train = np.array([i[:1] for i in y_train])
# y_train = np.where(y_train == 'A', 0, np.where(y_train == 'B', 1, np.where(y_train == 'C', 2, 3)))
#
# # Get y for test
# temp = test_data.copy()
# temp = temp.groupby('plant').sum()
# y_test = np.array(temp.index)
# y_test = np.array([i[:1] for i in y_test])
# y_test = np.where(y_test == 'A', 0, np.where(y_test == 'B', 1, np.where(y_test == 'C', 2, 3)))
#
# del train_data['plant']
# del test_data['plant']
#
# # Normalize train and test data
# data_norm_train = normalization(train_data)
# data_norm_test = normalization(test_data)
#
# # Reshape data to fit LSTM format
# train_temporal_data = data_norm_train.to_numpy().reshape(int(len(data_norm_train)/days), days, len(data_norm_train.columns))
# test_temporal_data = data_norm_test.to_numpy().reshape(int(len(data_norm_test)/days), days, len(data_norm_test.columns))
#
# # ------------------------------------------ Prepare data and train model -------------------------------------------- #
#
# Set parameters
train_days = 6
start_day = 1 + train_days
end_day = 41
plants_num = 188

# # Base LSTM parameters
# nb_units = 150
# batch_size = 8
# epochs = 150
# n_layers = 1
# activation = 'tanh'
# optimizer = 'SGD'
# l_rate = 0.001
# momentum = 0.99
#
# # Set performance measures
# Accuracies_train = np.zeros((end_day - start_day + 1, 1))
# Accuracies_test = np.zeros((end_day - start_day + 1, 1))
#
# # Loop through all days - train on previous 'train_days' and test on current day
# for day in range(start_day, end_day + 1):
#     # Get train and test data from the beginning until current day
#     curr_day = day - start_day + train_days
#     # X train and test data
#     train_x = train_temporal_data[:, :curr_day, :]
#     test_x = test_temporal_data[:, :curr_day, :]
#
#     # LSTM params
#     input_time_dim = train_x.shape[1]
#     input_seq_size = train_x.shape[2]
#
#     # Create the model
#     tf.keras.backend.clear_session()
#     Base_LSTM = Base_LSTM_Model(input_seq_size, input_time_dim, activation, optimizer, l_rate, momentum, nb_units, n_layers)
#
#     logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#
#     # Train model
#     Base_LSTM.fit(train_x, to_categorical(y_train),
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=1,
#                   callbacks=[tensorboard_callback])
#
#     # Predict train data with trained model
#     y_train_proba = Base_LSTM.predict(train_x)
#     y_train_pred = np.argmax(y_train_proba, axis=1)
#
#     # Predict test data with trained model
#     y_test_proba = Base_LSTM.predict(test_x)
#     y_test_pred = np.argmax(y_test_proba, axis=1)
#
#     # Save categorical performance measures of current day
#     train_acc = np.where(y_train == y_train_pred, 1, 0).sum() / len(y_train)
#     test_acc = np.where(y_test == y_test_pred, 1, 0).sum() / len(y_test)
#     Accuracies_train[day - start_day] = train_acc
#     Accuracies_test[day - start_day] = test_acc
#
#     print("Finished time step %s! Train Accuracy is: %s, Test Accuracy is: %s" % (day, train_acc, test_acc))
#
# # Save results categorical
# all_days = np.arange(start_day, end_day + 1)
# columns = ['Day', 'Accuracy_Train', 'Accuracy_Test']
# results = pd.DataFrame(columns=columns)
# results['Day'] = all_days
# results['Accuracy_Train'] = Accuracies_train
# results['Accuracy_Test'] = Accuracies_test
# results.to_csv(save_path + '/LSTM_Results_Final.csv', index=False)

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- Hyper-parameter Tuning With CV ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# ------------ Load Data ------------- #
path = r'C:\Users\mannh\Banana'
path_col = r'C:\Users\mannh\Banana'
save_path = r'C:\Users\mannh\Banana\Results'
df = pd.read_csv(path + '/All_features_smoothed.csv')
cols = pd.read_csv(path_col + '/Disc_Entropy.csv')
del cols['class']
cols = cols.columns

# Set params
plants = 47
change_point = 1
days = 41 - change_point + 1
end_day = 41
plants_num = 188

# ------------ Prepare Data ------------- #

# Remove stable days
df = df[df['Date'] >= change_point].reset_index()
df['plant'] = df['Treatment'] + df['Num'].astype(str)
del df['Treatment']
del df['Num']
del df['index']

# Get categorical Y
temp = df.copy()
temp = temp.groupby('plant').sum()
y_data = np.array(temp.index)
y_data = np.array([i[:1] for i in y_data])
y_data = np.where(y_data == 'D', 1, 0)
y_start = np.zeros(plants_num, int)

del df['plant']

df = df[cols]

# Normalize the data
data_norm = normalization(df)

# Reshape data to fit LSTM format
temporal_data = data_norm.to_numpy().reshape(int(len(data_norm) / days), days, len(data_norm.columns))

# --------------------------------------- Train and evaluate models with CV ------------------------------------------ #

# Base LSTM parameters
nb_units = [150]
batch_size = [8]
epochs = [150]
n_layers = [1]
train_days = [6]
activations = ['tanh']
optimizers = ['SGD']
l_rate = [0.001]
momentum = [0.99]
# Set performance measures
B_Accuracies_valid = np.zeros((end_day - start_day + 1, 1))
F1s_valid = np.zeros((end_day - start_day + 1, 1))
AUCs_valid = np.zeros((end_day - start_day + 1, 1))
items = []

# Number of parameters combinations to check
num_options = np.product([len(nb_units), len(batch_size), len(epochs), len(n_layers), len(train_days), len(activations),
                          len(optimizers)])
print("Number of parameters combinations is: ", num_options)

# Loop through every combination and get performance by CV
for ind, item in enumerate(itertools.product(nb_units, batch_size, n_layers, train_days, activations, optimizers, epochs, l_rate, momentum
                                             )):
    # The current parameters to check
    items.append(item)
    units = item[0]
    batch = item[1]
    layers = item[2]
    days = item[3]
    active = item[4]
    optimize = item[5]
    epoch = item[6]
    learning = item[7]
    moment = item[8]
    start_day = 1 + days
    accuracies = []

    # Loop through all days - using sequences of 'days' back
    for day in range(start_day, end_day + 1):
        # Get data from the beginning until current day
        curr_day = day - start_day + days
        valid_x = temporal_data[:, :curr_day, :]

        # LSTM params
        input_time_dim = valid_x.shape[1]
        input_seq_size = valid_x.shape[2]

        # Create the model
        tf.keras.backend.clear_session()

        logdir = os.path.join("../../Deep_Learning_Course/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        # Train model with CV
        if day < 14:
            scores = cv_LSTM(1, y_start, valid_x, y_data, batch, epoch, input_seq_size, input_time_dim, units, layers,
                             active, optimize, learning, moment)
        else:
            scores = cv_LSTM(0, y_start, valid_x, y_data, batch, epoch, input_seq_size, input_time_dim, units, layers,
                             active, optimize, learning, moment)
        b_accuracy = scores[0][0]
        f1_res = scores[1][0]
        auc_res = scores[2][0]
        # accuracies.append(accuracy)
        B_Accuracies_valid[day - start_day] = b_accuracy
        F1s_valid[day - start_day] = f1_res
        AUCs_valid[day - start_day] = auc_res
        print("Finished day: ", day)

    # Save results categorical
    all_days = np.arange(start_day, end_day + 1)
    columns = ['Day', 'Balanced_Accuracy', 'F1_pos', 'AUC']
    results = pd.DataFrame(columns=columns)
    results['Day'] = all_days
    results['Balanced_Accuracy'] = B_Accuracies_valid
    results['F1_pos'] = F1s_valid
    results['AUC'] = AUCs_valid
    results.to_csv(save_path + '/LSTM_CV_Results_fromstart_stressOnly_auc.csv', index=False)

