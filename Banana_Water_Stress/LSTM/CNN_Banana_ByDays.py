import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
import os
import numpy as np
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.utils import to_categorical


def load_images(path, day_start, day_end, jump):
    f_names = os.listdir(path)
    images = []
    for index, image in enumerate(f_names):
        if (index >= day_start * jump) & (index < day_end * jump):
            temp_image = cv2.imread(os.path.join(path, image), -1)
            images.append(temp_image)
    return images


def combine_channels(RGB, depth, thermal):
    images_combined = []
    for index in range(len(RGB)):
        temp_img = np.zeros((512, 512, 5))
        temp_img[:, :, 0:3] = RGB[index]
        temp_img[:, :, 3] = depth[index]
        temp_img[:, :, 4] = thermal[index]
        images_combined.append(temp_img)
    return images_combined


def create_label_binary(days_num, jump, plants):
    day_label = np.zeros((jump, 1), dtype=int)
    day_label[jump - plants:] = 1
    binary_y = []
    for i in range(0, days_num):
        binary_y = np.concatenate((binary_y, day_label), axis=None)
    return binary_y


def create_label_categories(days_num, jump, plants):
    day_label = np.zeros((jump, 1), dtype=int)
    day_label[plants:plants * 2] = 1
    day_label[plants * 2:plants * 3] = 2
    day_label[plants * 3:] = 3
    category_y = []
    for i in range(0, days_num):
        category_y = np.concatenate((category_y, day_label), axis=None)
    return category_y


def norm_images(images):
    images = (images - np.min(images)) / np.max(images)
    return images


def banana_model(X_train, Y_train):
    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(512, 512, 5)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # train the model
    model.fit(X_train, Y_train, epochs=3, verbose=1)
    return model


# ------------------------------------------ Prepare data and train model -------------------------------------------- #

# Set paths
# RGB_path = 'C:/DeepLearning/RGB_Masked'
# depth_path = 'C:/DeepLearning/Depth_Masked'
# thermal_path = 'C:/DeepLearning/Thermal_Masked'
RGB_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/RGB_Masked_0'
depth_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/Depth_Masked_0'
thermal_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/Thermal_Masked_m1'
save_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Deep_Learning_Course'

# Set parameters
train_days = 2
start_day = 14 + train_days
end_day = 41
plants_num = 192
categorical_plants_num = 48
train_size = 0.75
threshold = 0.3

# Set performance measures
Accuracies = np.zeros((end_day - start_day + 1, 1))
AUCs = np.zeros((end_day - start_day + 1, 1))
B_Accuracies = np.zeros((end_day - start_day + 1, 1))
F1s = np.zeros((end_day - start_day + 1, 1))
Classification_Reports = []

# Create labels binary
# y_train = create_label_binary(train_days, plants_num, category_plants_num)
# y_test = create_label_binary(1, plants_num, category_plants_num)

# Create labels categorical
y_train = create_label_categories(train_days, plants_num, categorical_plants_num)
y_test = create_label_categories(1, plants_num, categorical_plants_num)

# Transform y to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Loop through all days - train on previous 'train_days' and test on current day
for day in range(start_day, end_day + 1):
    # Load train days images (past 'train_days' from current day)
    RGB_train = load_images(RGB_path, day - train_days - 1, day - 1, plants_num)
    depth_train = load_images(depth_path, day - train_days - 1, day - 1, plants_num)
    thermal_train = load_images(thermal_path, day - train_days - 1, day - 1, plants_num)

    # Load test days images (current day)
    RGB_test = load_images(RGB_path, day - 1, day, plants_num)
    depth_test = load_images(depth_path, day - 1, day, plants_num)
    thermal_test = load_images(thermal_path, day - 1, day, plants_num)

    # Normalize train and test images to [0, 1]
    RGB_images_train = norm_images(RGB_train)
    depth_images_train = norm_images(depth_train)
    thermal_images_train = norm_images(thermal_train)

    RGB_images_test = norm_images(RGB_test)
    depth_images_test = norm_images(depth_test)
    thermal_images_test = norm_images(thermal_test)

    # Combine train and test images
    combined_images_train = combine_channels(RGB_images_train, depth_images_train, thermal_images_train)
    combined_images_test = combine_channels(RGB_images_test, depth_images_test, thermal_images_test)

    # Convert to integers
    x_train = np.array(combined_images_train, dtype=float)
    x_test = np.array(combined_images_test, dtype=float)
    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)

    # Train model
    model_banana = banana_model(x_train, y_train)

    # Predict with trained model
    probabilities = model_banana.predict(x_test)
    # predictions = np.where(probabilities[:, 1] >= threshold, 1, 0)
    # predictions = np.where(probabilities >= threshold, 1, 0)
    predictions = np.argmax(probabilities, axis=1)    # For categorical predictions

    # Save binary performance measures of current day
    # fpr, tpr, thresholds = roc_curve(y_test[:, 1], probabilities[:, 1], pos_label=1)
    # AUCs[day - start_day] = auc(fpr, tpr)
    # B_Accuracies[day - start_day] = balanced_accuracy_score(y_test[:, 1], predictions)
    # F1s[day - start_day] = f1_score(y_test[:, 1], predictions)
    # ClassificatSion_Reports.append(classification_report(y_test[:, 1], predictions))
    # print(Classification_Reports[day - start_day])

    # Save categorical performance measures of current day
    Accuracies[day - start_day] = accuracy_score(y_test, to_categorical(predictions))


# Save results binary
all_days = np.arange(start_day, end_day + 1)
# columns = ['Day', 'AUC', 'B_Accuracy', 'F1']
# results = pd.DataFrame(columns=columns)
# results['Day'] = all_days
# results['AUC'] = AUCs
# results['B_Accuracy'] = B_Accuracies
# results['F1'] = F1s
# results.to_csv(save_path + '/Results_binary.csv', index=False)

# Save results categorical
columns = ['Day', 'Accuracy']
results = pd.DataFrame(columns=columns)
results['Day'] = all_days
results['Accuracy'] = Accuracies
results.to_csv(save_path + '/Results_categorical.csv', index=False)


