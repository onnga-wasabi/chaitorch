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
from utils.updater import Updater
from utils.models import LeNet
from utils.extension import (
    LogReport,
    ProgressBar,
    ClassifyEvaluater,
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

    loss_fn = nn.CrossEntropyLoss()

    updater = Updater(net, loss_fn, device, optim='Adam')
    trainer = Trainer(5, updater, train_data_loader)
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        'training/accuracy',
        'validation/loss',
        'validation/accuracy',
        'elapsed_time',
    ], {'epoch': 1}))
    trainer.extend(ProgressBar(10))
    trainer.extend(ClassifyEvaluater(test_data_loader))
    trainer.run()


if __name__ == '__main__':
    main()
