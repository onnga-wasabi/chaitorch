import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import (
    data,
)
from torchvision import (
    datasets,
    transforms,
    models,
)
from fastprogress import (
    progress_bar,
    master_bar,
)
import cv2


data_dir = './data'


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Cifartrfm(object):

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, image):
        print(image.size)
        return cv2.resize(image, self.output_size)


def setLossAndOptimizer(net):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    return loss, optimizer


def main():

    device = torch.device('cpu')
    # device = torch.device('cuda:0')

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=data_transform,
        download=True,
    )
    train_data_loader = data.DataLoader(train_dataset, batch_size=64)

    # net = models.resnet18(pretrained=True)
    # net = models.resnet18(num_classes=10)
    net = LeNet().to(device)
    loss_fn, optimizer = setLossAndOptimizer(net)

    epochs = 2
    mb = master_bar(range(epochs))
    for epoch in mb:
        current_loss = 0
        corrects = 0
        for i, pair in zip(progress_bar(range(len(train_data_loader.dataset)), parent=mb), train_data_loader):
            optimizer.zero_grad()

            pair = pair.to(device)
            images, labels = pair
            out = net(images)
            loss = loss_fn(out, labels)
            loss.backward()
            _, preds = torch.max(out, 1)
            optimizer.step()

            current_loss += loss.item() * images.shape[0]
            corrects += torch.sum(preds == labels)

        epoch_loss = current_loss / len(train_dataset)
        acc = corrects.double() / len(train_dataset)
        print(f'{epoch} Loss: {epoch_loss:.4f} ACC: {acc:.4f}')


if __name__ == '__main__':
    main()
