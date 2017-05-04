# original:  InfoGAN/infogan/misc/datasets.py
# modified: Ekaterina Sutter, May 2017

# from tensorflow.examples.tutorials import mnist
import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np


class Dataset(object):
    def __init__(self, dataloader):

        self.dataloader = list(dataloader)

        self._numiter = len(dataloader)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def num_examples(self):
        raise NotImplementedError

    @property
    def epochs_size(self):
        return self._numiter

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self):
        """Return the next batch."""
        data, target = self.dataloader[self._index_in_epoch]
        data = data.view(-1, 1, 28*28)

        self._index_in_epoch += 1

        if self._index_in_epoch == self._numiter-1:
            self._epochs_completed += 1
            self._index_in_epoch = 0

        return data, target


class MnistDataset(object):
    def __init__(self, batch_size):

        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST(data_directory, train=True, download=False,
                                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                 # transforms.Normalize((0.1307,), (0.3081,))
                                                                                     ])),
                            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(data_directory, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False)

        self.train = Dataset(train_loader)  # dataset.train
        self.test = Dataset(test_loader)  # dataset.test

        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
