import torch
import torch.nn as nn


class BananaNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BananaNet, self).__init__()
        self.model1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(3, 3))
        self.model2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.model3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.model4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.model5 = nn.Dropout2d(p=0.2)
        self.model6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.model7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3))
        self.model8 = nn.MaxPool2d(kernel_size=(2, 2))
        self.model9 = nn.Dropout2d(p=0.2)
        self.model10 = nn.AvgPool2d(kernel_size=(3, 3))
        self.model11 = nn.Flatten()
        self.model12 = nn.Linear(in_features=107584, out_features=64)
        self.model13 = nn.Linear(in_features=64, out_features=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_l):
        conv = self.model1(input_l)
        conv = self.activation(conv)
        conv = self.model2(conv)
        conv = self.activation(conv)
        conv = self.model3(conv)
        conv = self.activation(conv)
        conv = self.model4(conv)
        conv = self.model5(conv)
        conv = self.model6(conv)
        conv = self.activation(conv)
        conv = self.model7(conv)
        conv = self.activation(conv)
        conv = self.model8(conv)
        conv = self.model9(conv)
        conv = self.model10(conv)
        conv = self.model11(conv)
        conv = self.model12(conv)
        conv = self.activation(conv)
        conv = self.model13(conv)
        out_reg = self.sigmoid(conv)
        return out_reg


def YotaMit(Pretraind=False):
    model = BananaNet()
    return model
