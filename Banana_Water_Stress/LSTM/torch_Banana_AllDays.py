import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import torch.optim as optim
from BananaNet import *
from torch.utils.data import DataLoader, TensorDataset


def load_images(path, days_num, jump):
    f_names = os.listdir(path)
    images = []
    for index, image in enumerate(f_names):
        if index < days_num * jump:
            temp_image = np.asarray(Image.open(os.path.join(path, image)))
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
    stress_start = jump - stress_num  # 144
    train_healthy_end = int((jump - 48) * train_percentage)  # 108
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


def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield start, x
        start += step


# ------------------------------------------ Prepare data and train model -------------------------------------------- #

# Set paths
home = os.getcwd()
RGB_path = os.path.join(home, 'RGB_Masked')
depth_path = os.path.join(home, 'Depth_Masked')
thermal_path = os.path.join(home, 'Thermal_Masked')

# Set parameters
days = 1
plants_num = 192
categorical_plants_num = 48
train_size = 0.75
threshold = 0.3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
# y = create_label_categories(days, plants_num, categorical_plants_num)

# Split data
[x_train, x_test, y_train, y_test] = split_data(combined_images, y, days, plants_num, train_size)

# Convert to integers
x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)
y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)

# Convert to Tensor
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Create batches and connect x to labels
batch_size = 2  # jump 2^x
data_set = TensorDataset(x_train, y_train)
trainset = DataLoader(data_set, batch_size=batch_size, shuffle=True)
data_test = TensorDataset(x_test, y_test)
testset = DataLoader(data_test, batch_size=batch_size, shuffle=True)


# Transform y to categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# Build model
net = YotaMit().eval()
# net.cuda()
net.double()  # conv2d work with double
criterion = nn.BCELoss()  # binary cross entropy
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  ##### need to be changed to Adam

# Train model
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainset):
        # get the inputs; data is a list of [inputs, labels] and reshape them to net
        inputs, labels = data
        inputs = inputs.reshape(batch_size, 5, 512, 512)
        labels = labels.type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.squeeze(1)
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        print(loss, i)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
