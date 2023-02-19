import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the
        # 68 keypoint (x, y) pairs

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(256*12*12, 4096)
        self.drop5 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(4096, 512)
        self.drop6 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(512, 136)

    def forward(self, x):
        # 1x224x224
        x = F.relu(self.conv1(x))
        # 32x222x222
        x = self.bn1(x)
        # 32x222x222
        x = self.pool1(x)
        # 32x111x111
        x = self.drop1(x)
        # 32x111x111

        # 32x111x111
        x = F.relu(self.conv2(x))
        # 64x109x109
        x = self.bn2(x)
        # 64x109x109
        x = self.pool2(x)
        # 64x54x54
        x = self.drop2(x)
        # 64x54x54

        # 64x54x54
        x = F.relu(self.conv3(x))
        # 128x52x52
        x = self.bn3(x)
        # 128x52x52
        x = self.pool3(x)
        # 128x26x26
        x = self.drop3(x)
        # 128x26x26

        # 128x26x26
        x = F.relu(self.conv4(x))
        # 256x24x24
        x = self.bn4(x)
        # 256x24x24
        x = self.pool4(x)
        # 256x12x12
        x = self.drop4(x)

        # 256x12x12 (flatten)
        x = x.view(x.size(0), -1)
        # 256*12*12
        x = F.relu(self.fc1(x))
        # 4096
        x = self.drop5(x)
        # 4096
        x = F.relu(self.fc2(x))
        # 512
        x = self.drop6(x)
        # 512
        x = self.fc3(x)
        # 136

        return x


class NetNG(nn.Module):

    def __init__(self):
        super(NetNG, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the
        # 68 keypoint (x, y) pairs

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc_drop = nn.Dropout(p=0.2)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*12*12, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 136)

    def forward(self, x):
        # 1x224x224
        x = F.relu(self.conv1(x))
        # 32x222x222
        x = self.pool(x)
        # 32x111x111
        x = self.dropout(x)
        # 32x111x111
        x = self.batch_norm1(x)

        # 32x111x111
        x = F.relu(self.conv2(x))
        # 64x109x109
        x = self.pool(x)
        # 64x54x54
        x = self.dropout(x)
        # 64x54x54
        x = self.batch_norm2(x)

        # 64x54x54
        x = F.relu(self.conv3(x))
        # 128x52x52
        x = self.pool(x)
        # 128x26x26
        x = self.dropout(x)
        # 128x26x26
        x = self.batch_norm3(x)

        # 128x26x26
        x = F.relu(self.conv4(x))
        # 256x24x24
        x = self.pool(x)
        # 256x12x12
        x = self.dropout(x)
        # 256x12x12
        x = self.batch_norm4(x)
        # 256x12x12 (flatten)
        x = x.view(x.size(0), -1)

        # 36864
        x = F.relu(self.fc1(x))
        # 4096
        x = self.fc_drop(x)
        # 4096
        x = F.relu(self.fc2(x))
        # 512
        x = self.fc_drop(x)
        # 512
        x = self.fc3(x)
        # 136

        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        # Output = (32, 220, 220)
        # Maxpooled output = (32, 110, 110)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.2)
        # output = (64, 108, 108)
        # Maxpooled output = (64, 54, 54)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.3)
        # Output = (128, 52, 52)
        # Maxpooled Output = (128, 26, 26)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.4)
        # Output = (256, 24, 24)
        # Maxpooled Output = (256, 12, 12)

        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0.5)
        # Output = (512, 10, 10)
        # Maxpooled Output = (512, 5, 5)

        self.fc6 = nn.Linear(512*5*5, 2560)
        self.drop6 = nn.Dropout(p=0.4)

        self.fc7 = nn.Linear(2560, 1280)
        self.drop7 = nn.Dropout(p=0.4)

        self.fc8 = nn.Linear(1280, 136)

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)

        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)

        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = self.drop6(x)

        x = F.relu(self.fc7(x))
        x = self.drop7(x)

        x = self.fc8(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
