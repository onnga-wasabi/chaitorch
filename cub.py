import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils import (
    data,
)
from torchvision import (
    transforms,
    models,
)

import chaitorch
from chaitorch.training.trainer import Trainer
from chaitorch.training.updater import Updater
from chaitorch.training.trigger import MinValueTrigger
from chaitorch.training.extension import (
    LogReport,
    ProgressBar,
    ClassifyEvaluater,
    SnapshotModel,
)


DATA_DIR = './data'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    return parser.parse_args()


def main():
    timestamp = datetime.now().strftime('%y-%m-%d/%H%M%S')

    args = parser()

    if args.gpu > -1:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=True,
        transform=data_transform,
        download=True,
    )
    test_dataset = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=False,
        transform=data_transform,
        download=True,
    )
    train_data_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, batch_size=64)

    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 200)

    updater = Updater(net, train_data_loader, device, optim='Adam', lr_=1e-5)
    trainer = Trainer(updater, {'epoch': 1})
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        'training/accuracy',
        'validation/loss',
        'validation/accuracy',
        'elapsed_time',
    ], {'epoch': 1}))
    trainer.extend(ProgressBar(30))
    trainer.extend(ClassifyEvaluater(test_data_loader))
    trigger = MinValueTrigger('validation/loss')
    trainer.extend(SnapshotModel(timestamp, trigger))

    trainer.run()


if __name__ == '__main__':
    main()
