from torch import nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):

    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        h = F.relu(F.max_pool2d(self.conv1(x), 2))
        h = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(h)), 2))
        h = h.view(-1, 320)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, training=self.training)
        h = self.fc2(h)
        return F.normalize(h, p=2, dim=1)


class FinetuneCNN(nn.Module):

    def __init__(self):
        super(FinetuneCNN, self).__init__()
        self.features = None

    def forward(self, x):
        h = self.features(x)
        h = F.avg_pool2d(h, 7)
        return F.normalize(h, p=2, dim=1)
