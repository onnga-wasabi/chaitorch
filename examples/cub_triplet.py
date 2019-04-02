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
    MetricEvaluater,
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset_core = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=True,
        transform=data_transform,
        download=True,
        mode='zero-shot',
    )
    train_dataset = TripletDataset(train_dataset_core)

    test_dataset_core = chaitorch.utils.datasets.CUB2002011(
        root=DATA_DIR,
        train=False,
        transform=data_transform,
        download=True,
        mode='zero-shot',
    )
    test_dataset = TripletDataset(test_dataset_core)

    train_data_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_data_loader = data.DataLoader(test_dataset, batch_size=32, num_workers=8)

    # base_net = models.vgg16(pretrained=True)
    # net = models.vgg16(pretrained=False)
    # net.features = base_net.features
    # net.classifier = nn.Sequential(
    #     nn.Linear(512 * 7 * 7, 4096),
    # )
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, 512)

    updater = TripletLossUpdater(net, train_data_loader, device=device, optim='Adam', lr_=1e-5)
    trainer = Trainer(updater, {'epoch': 50}, out=f'result/{timestamp}')
    trainer.extend(LogReport([
        'epoch',
        'training/loss',
        'eval/loss',
        'eval/R@1',
        'eval/R@2',
        'eval/R@4',
        'eval/R@8',
        'elapsed_time',
    ], {'epoch': 1}))
    trainer.extend(ProgressBar(30))
    trainer.extend(MetricEvaluater(test_data_loader))
    # trigger = MinValueTrigger('eval/loss')
    # trainer.extend(SnapshotModel(trainer.out, trigger=trigger))

    trainer.run()


if __name__ == '__main__':
    main()
