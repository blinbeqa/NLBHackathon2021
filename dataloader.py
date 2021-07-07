

import os, glob
import numpy as np

import random

from torch.utils.data.dataset import Dataset


# Dataloder for all Moving-MNIST datasets (binary and colored)

class NLB_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset
        path = params['path']
        assert os.path.exists(path), "The dataset folder does not exist."

        self.data = np.load(path)


        self.data_samples = len(self.data)

    def __getitem__(self, index):

        data = self.data[index]
        return data

    def __len__(self):
        return self.data_samples
