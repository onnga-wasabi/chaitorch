import numpy as np


class TripletDataset(object):

    def __init__(self, dataset):
        self.dataset = dataset
        # converting for MNIST
        self.labels = [int(label) for label in dataset.targets]
        unique_labels = set(self.labels)
        self.label_idx_dict = {}
        for label in unique_labels:
            self.label_idx_dict[label] = [i for i, x in enumerate(self.labels) if x == label]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x_a, anchor_label = self.dataset[index]

        positive_label_idx = np.random.choice(self.label_idx_dict[anchor_label])
        x_p, _ = self.dataset[positive_label_idx]

        negative_label = np.random.choice([k for k in self.label_idx_dict.keys() if k != anchor_label])
        negative_label_idx = np.random.choice(self.label_idx_dict[negative_label])
        x_n, _ = self.dataset[negative_label_idx]

        return x_a, x_p, x_n
