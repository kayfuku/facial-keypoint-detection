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
        # 256x12x12
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
