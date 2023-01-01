import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import *
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import itertools
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 55)


# Define the base model
def get_base_model(n_features, size_layers, active, solve, dropping):
    # define model
    base_model = Sequential()
    base_model.add(Dense(size_layers, input_dim=n_features, activation=active, kernel_initializer='he_uniform'))
    base_model.add(Dropout(dropping))
    base_model.add(Dense(size_layers, activation=active, kernel_initializer='he_uniform'))
    base_model.add(Dropout(dropping))
    base_model.add(Dense(size_layers, activation=active, kernel_initializer='he_uniform'))
    base_model.add(Dense(4, activation='softmax'))

    # compile model
    if solve == 'sgd':
        opt = SGD(learning_rate=0.01, momentum=0.9)
    if solve == 'adam':
        opt = Adam(learning_rate=0.01)
    base_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return base_model


def prepare_data(data, train_days, test_day, columns):
    # Split to train and test
    data_train = data[data['Date'].isin(train_days)]
    data_test = data[data['Date'] == test_day]
    y_train = np.where(data_train['Treatment'] == 'A', 0, np.where(data_train['Treatment'] == 'B', 1, np.where(data_train['Treatment'] == 'C', 2, 3)))
    y_test = np.where(data_test['Treatment'] == 'A', 0, np.where(data_test['Treatment'] == 'B', 1, np.where(data_test['Treatment'] == 'C', 2, 3)))
    # Transform y to categorical
    y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # If need to choose only selected features
    # data_train = data_train[columns]
    # data_test = data_test[columns]

    # Hide this if selected columns are used
    del data_train['Treatment']
    del data_train['Date']
    del data_train['Num']

    del data_test['Treatment']
    del data_test['Date']
    del data_test['Num']

    # Scale data
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.fit_transform(data_test)
    return data_train, y_train, data_test, y_test


path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
path_col = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\Discretization'
save_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Deep_Learning_Course'
df = pd.read_csv(path + '/All_features_smoothed.csv')
cols = pd.read_csv(path_col + '/Disc_Entropy.csv')
del cols['class']
cols = cols.columns

threshold = 0.5
days_train = 2
days = 41
change_point = 14
time_steps = range(1 + days_train, days + 1, 1)
plot = 0
save = 1
printing = 1
stress_accuracies = []
items = []

# hidden_layer_sizes = np.array(range(70, 200, 30))
# activations = ['relu', 'tanh']
# solvers = ['adam', 'sgd']
# epochs = np.array(range(100, 201, 50))
# dropout = np.arange(0, 0.25, 0.1)

hidden_layer_sizes = [160]
activations = ['relu']
solvers = ['sgd']
epochs = [200]
dropout = [0.1]

num_options = np.product([len(hidden_layer_sizes), len(activations), len(solvers), len(epochs), len(dropout)])
print("Number of parameters combinations is: ", num_options)

for ind, item in enumerate(itertools.product(hidden_layer_sizes, activations, solvers, epochs, dropout)):

    items.append(item)
    hidden = item[0]
    activation = item[1]
    solver = item[2]
    epoch = item[3]
    drop = item[4]
    accuracies = []

    # Loop through all days - train on past 'days_train' days and test the current day
    for i in range(days_train + 1, days + 1):
        train_data, train_y, test_data, test_y = prepare_data(df, range(i - days_train, i), i, cols)

        # Create the base model for the current day's data
        model = get_base_model(train_data.shape[1], hidden, activation, solver, drop)
        # Train the base model
        model.fit(train_data, train_y, epochs=epoch, verbose=2)
        # Get predictions for test data
        probabilities = model.predict(test_data, verbose=1)
        predictions = np.argmax(probabilities, axis=1)
        predictions = predictions.reshape(1, -1)
        # Compute current accuracy and append it to all days accuracies
        accuracy = np.where(predictions == test_y, 1, 0)
        accuracy = np.sum(accuracy) / accuracy.shape[1]
        accuracies.append(accuracy)
        print("Test accuracy for day", i, " is: ", np.round(accuracy, 4))

    stress_accuracy = round(np.mean(accuracies[change_point:]), 4)
    stress_accuracies.append(stress_accuracy)

    if printing:
        print("Average accuracy of all days is: ", round(np.mean(accuracies), 4))
        print("Average accuracy of stable days is: ", round(np.mean(accuracies[:change_point]), 4))
        print("Average accuracy of only stress days is: ", round(np.mean(accuracies[change_point:]), 4))

    if plot:
        plt.plot(accuracies)

    if save:
        results = pd.DataFrame(np.concatenate([np.array(time_steps).reshape(-1, 1), np.array(accuracies).reshape(-1, 1)], axis=1))
        results.columns = ['Time_step', 'Accuracy']
        results.to_csv(save_path + '/Results_categorical_ANN.csv', index=False)

    print("Finished checking param combination number %s - %s" % (ind, item))

    tuning_results = pd.DataFrame(list(zip(items, stress_accuracies)))
    tuning_results.columns = ['parameters', 'stress_accuracy']
    tuning_results.to_csv(save_path + '/Results_tuning2.csv', index=False)

print(tuning_results)


# Tuning:
# Tuning_ANN_model = MLPClassifier(random_state=28)
#
# parameters = {"hidden_layer_sizes": range(40, 200, 30),
#               "activation": ['relu', 'logistic', 'tanh'],
#               "solver": ['adam', 'sgd'],
#               "max_iter": range(100, 300, 50)}
# grid_search = GridSearchCV(estimator=Tuning_ANN_model,
#                            param_grid=parameters,
#                            cv=k_folds,
#                            return_train_score=True,
#                            verbose=True,
#                            scoring='roc_auc')
# grid_search.fit(x_train, y_train)
# print("Best: ", grid_search.best_estimator_)


