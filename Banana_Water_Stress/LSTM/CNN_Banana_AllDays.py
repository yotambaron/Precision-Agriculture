import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical


def load_images(path, days_num, jump):
    f_names = os.listdir(path)
    images = []
    for index, image in enumerate(f_names):
        if index >= (41 - days_num) * jump:
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
    binary_y = []
    for i in range(0, days_num):
        binary_y = np.concatenate((binary_y, day_label), axis=None)
    return binary_y


# def normalize_images(images):
#     for img in range(len(images)):
#         for channel in range(np.shape(images[img])[2]):
#             images[img][:, :, channel] = \
#                 (images[img][:, :, channel] - np.min(images[img][:, :, channel])) / np.max(images[img][:, :, channel])
#     return images


def norm_images(images):
    images = (images - np.min(images)) / np.max(images)
    return images


def split_data(images_combined, labels, days_num, jump, train_percentage):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    stress_num = int(jump / 4)  # 48
    stress_start = jump - stress_num    # 144
    train_healthy_end = int((jump - 48) * train_percentage)   # 108
    train_stress_end = int(stress_start + stress_num * train_percentage)  # 180

    for day in range(0, days_num):
        day_images = images_combined[day * jump: (day + 1) * jump]
        temp_y = labels[day * jump: (day + 1) * jump]
        # train healthy images (0 - 107)
        for im in range(0, train_healthy_end):
            train_x.append(day_images[im])
            train_y.append(temp_y[im])
        # test healthy images (108 - 143)
        for im in range(train_healthy_end, stress_start):
            test_x.append(day_images[im])
            test_y.append(temp_y[im])
        # train stress images (144 - 179)
        for im in range(stress_start, train_stress_end):
            train_x.append(day_images[im])
            train_y.append(temp_y[im])
        # test stress images (180 - 191)
        for im in range(train_stress_end, jump):
            test_x.append(day_images[im])
            test_y.append(temp_y[im])
    return train_x, test_x, train_y, test_y


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
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # train the model
    model.fit(X_train, Y_train, epochs=1, verbose=1)
    return model


# ------------------------------------------ Prepare data and train model -------------------------------------------- #

# Set paths
# RGB_path = 'C:/DeepLearning/RGB_Masked'
# depth_path = 'C:/DeepLearning/Depth_Masked'
# thermal_path = 'C:/DeepLearning/Thermal_Masked'
RGB_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/RGB_Masked'
depth_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/Depth_Masked'
thermal_path = 'C:/Users/yotam/Desktop/yotam/Phenomix/EX6_WaterStress/Images/same_size_pics/Thermal_Masked'

# Set parameters
days = 1
plants_num = 192
categorical_plants_num = 48
train_size = 0.75
threshold = 0.3

# Load images
RGB_images = load_images(RGB_path, days, plants_num)
depth_images = load_images(depth_path, days, plants_num)
thermal_images = load_images(thermal_path, days, plants_num)

# Normalize images to [0, 1]
RGB_images = norm_images(RGB_images)
depth_images = norm_images(depth_images)
thermal_images = norm_images(thermal_images)

# Combine images
combined_images = combine_channels(RGB_images, depth_images, thermal_images)

# Create labels
y = create_label_binary(days, plants_num, categorical_plants_num)
y = create_label_categories(days, plants_num, categorical_plants_num)

# Split data
[x_train, x_test, y_train, y_test] = split_data(combined_images, y, days, plants_num, train_size)

# Convert to integers
x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)
y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)

# Transform y to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Train model
model_banana = banana_model(x_train, y_train)

# Predict with trained model
probabilities = model_banana.predict(x_test)
predictions = np.where(probabilities[:, 1] >= threshold, 1, 0)
# predictions = np.where(probabilities >= threshold, 1, 0)
# predictions = np.argmax(probabilities, axis=0)    # For categorical predictions

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
