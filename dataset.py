from __future__ import absolute_import
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


# training : validation : testing
_RATIO = (70, 10, 20)

class MNISTDataset(data.Dataset):
    """dataset for bouncing MNIST"""
    def __init__(self, data_path):
        super(MNISTDataset, self).__init__()
        assert os.path.exists(data_path)
        self.data = np.load(data_path)
        print('dataset shape:', self.data.shape)

        self.vid_len, self.num_videos, self.image_size, _ = self.data.shape

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        # self.vid_len x self.image_size x self.image_size
        return torch.from_numpy(self.data[:, index, :, :]).long()

    @staticmethod
    def show_video(video):
        pass


if __name__ == '__main__':
    data_path = 'data/mnist_test_seq.npy'
    dataset = MNISTDataset(data_path)
    vid = dataset[0]
    print(vid.size())

    loader = data.DataLoader(dataset, batch_size=2)
    vids = next(iter(loader))
    print(vids.size())