import argparse

import torch
import torch.nn as nn
from torch.utils import (
    data,
)
from torchvision import (
    datasets,
    transforms,
    models,
)

from utils.trainer import Trainer
from utils.updater import StandardUpdater
from utils.models import LeNet
from utils.extension import (
    LogReport,
    ProgressBar,
)


DATA_DIR = './data'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    return parser.parse_args()


def main():

    args = parser()

    if args.gpu > -1:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    data_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
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
    net = LeNet()
    # net = net.to(device)
    # net = models.resnet18(num_classes=10).to(device)
    # net = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    updater = StandardUpdater(net, optimizer, device, loss_fn)
    trainer = Trainer(2, updater, train_data_loader)
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        'training/accuracy',
        'elapsed_time',
    ], {'epoch': 1}))
    trainer.extend(ProgressBar(10))
    trainer.run()

    '''
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
    '''


if __name__ == '__main__':
    main()
