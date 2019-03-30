import os
import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision.datasets.utils import download_url

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CUB2002011(data.Dataset):
    base_folder = 'CUB_200_2011'
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    images_folder = 'images'
    images_txt = 'images.txt'
    labels_txt = 'image_class_labels.txt'
    split_txt = 'train_test_split.txt'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, mode='normal'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        images_path = os.path.join(self.root, self.base_folder, self.images_txt)
        with open(images_path, 'rt') as rf:
            all_data = []
            for line in rf.read().strip().split('\n'):
                all_data.append(os.path.join(self.root, self.base_folder, self.images_folder, line.split(' ')[1]))

        labels_path = os.path.join(self.root, self.base_folder, self.labels_txt)
        with open(labels_path, 'rt') as rf:
            all_targets = [int(line.split(' ')[1]) - 1 for line in rf.read().strip().split('\n')]

        if mode == 'normal':
            split_path = os.path.join(self.root, self.base_folder, self.split_txt)
            with open(split_path, 'rt') as rf:
                mask = [int(line.split(' ')[1]) for line in rf.read().strip().split('\n')]
                if train:
                    idx = [x == 0 for x in mask]
                else:
                    idx = [x == 1 for x in mask]

        elif mode == 'zero-shot':
            ZERO_SHOT_TRAIN = 5864
            if train:
                idx = range(ZERO_SHOT_TRAIN)
            else:
                idx = range(ZERO_SHOT_TRAIN, len(all_data) + 1)

        else:
            print('mode ha normal ka zero-shot desu.')
            import sys
            sys.exit()

        self.data = np.array(all_data)[idx]
        self.targets = np.array(all_targets)[idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import tarfile

        download_url(self.url, self.root, self.filename)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)
