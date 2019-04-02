import argparse

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


DATA_DIR = './data'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', 'gpu', type=int, default=-1)
    return parser.parse_args()


def main():

    args = parser()

    if args.gpu > -1:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=data_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=data_transform,
        download=True,
    )
    train_data_loader = data.DataLoader(train_dataset, batch_size=64)
    test_data_loader = data.DataLoader(test_dataset, batch_size=64)

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 10)
    net = net.to(device)
    # net = models.resnet18(num_classes=10).to(device)
    # net = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    epochs = 2

    trainer 

    mb = master_bar(range(epochs))
    for epoch in mb:
        current_loss = 0
        corrects = 0
        for i, pair in zip(progress_bar(range(len(train_data_loader)), parent=mb), train_data_loader):
            optimizer.zero_grad()

            images, labels = pair
            images = images.to(device)
            labels = labels.to(device)
            out = net(images)
            loss = loss_fn(out, labels)
            loss.backward()
            _, preds = torch.max(out, 1)
            optimizer.step()

            current_loss += loss.item() * images.shape[0]
            corrects += torch.sum(preds == labels)

        epoch_loss = current_loss / len(train_dataset)
        acc = corrects.double() / len(train_dataset)

        val_corrects = 0
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                out = net(images)
                _, preds = torch.max(out, 1)
                val_corrects += torch.sum(preds == labels)
                val_acc = val_corrects.double() / len(test_dataset)

        print(f'{epoch} Loss: {epoch_loss:.4f} ACC: {acc:.4f} VALACC: {val_acc:.4f}')


if __name__ == '__main__':
    main()
