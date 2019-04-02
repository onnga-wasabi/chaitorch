import argparse
from datetime import datetime

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

from chaitorch.data.dataset import TripletDataset
from chaitorch.training.trainer import Trainer
from chaitorch.training.updater import TripletLossUpdater
from chaitorch.training.trigger import MinValueTrigger
from chaitorch.training.extension import (
    LogReport,
    ProgressBar,
    SnapshotModel,
    MetricEvaluater,
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
    train_dataset_core = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=data_transform,
        download=True,
    )
    train_dataset = TripletDataset(train_dataset_core)
    test_dataset_core = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=data_transform,
        download=True,
    )
    test_dataset = TripletDataset(test_dataset_core)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, num_workers=8)
    test_data_loader = data.DataLoader(test_dataset, batch_size=64, num_workers=8)

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 512)

    updater = TripletLossUpdater(net, train_data_loader, device, optim='Adam', lr_=1e-3)
    trainer = Trainer(updater, {'epoch': 1}, out=f'result/{timestamp}')
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        'eval/loss',
        'eval/R@1',
        'eval/R@2',
        'eval/R@4',
        'eval/R@8',
        'elapsed_time',
    ], {'iteration': 10}))
    trainer.extend(ProgressBar(50))
    trainer.extend(MetricEvaluater(test_data_loader))
    trigger = MinValueTrigger('eval/loss')
    trainer.extend(SnapshotModel(timestamp, trigger))

    trainer.run()


if __name__ == '__main__':
    main()
