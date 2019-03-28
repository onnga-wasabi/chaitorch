import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils import (
    data,
)
import torchvision
from torchvision import (
    transforms,
    models,
)

import chaitorch
from chaitorch.training.trainer import Trainer
from chaitorch.training.trigger import MinValueTrigger
from chaitorch.training.extension import (
    LogReport,
    ProgressBar,
    ClassifyEvaluater,
    SnapshotModel,
)
from chaitorch.data.dataset import TripletDataset
from chaitorch.training.updater import TripletLossUpdater

from model import FinetuneCNN


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
    ])

    train_dataset_core = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=True,
        transform=data_transform,
        download=True,
    )
    train_dataset = TripletDataset(train_dataset_core)

    test_dataset_core = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=False,
        transform=data_transform,
        download=True,
    )
    test_dataset = TripletDataset(test_dataset_core)

    train_data_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, batch_size=16)

    base_net = models.vgg16_bn(pretrained=True)
    # base_net = models.resnet18(pretrained=True)
    net = FinetuneCNN()
    net.features = base_net.features

    updater = TripletLossUpdater(net, train_data_loader, device=device, optim='Adam', lr_=1e-3)
    trainer = Trainer(updater, {'epoch': 1})
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        # 'training/accuracy',
        'validation/loss',
        # 'validation/accuracy',
        'elapsed_time',
    ], {'epoch': 1}))
    trainer.extend(ProgressBar(30))
    trainer.extend(ClassifyEvaluater(test_data_loader))
    trigger = MinValueTrigger('validation/loss')
    trainer.extend(SnapshotModel(timestamp, trigger))

    trainer.run()


if __name__ == '__main__':
    main()
