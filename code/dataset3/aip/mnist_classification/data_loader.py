import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tf
import pickle5 as pickle
import random
import os
import numpy as np
import tensorflow
   

class MnistDataset(Dataset):

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            x = x.view(-1)

        return x, y

from torchvision import datasets, transforms

dataset = datasets.MNIST(
    '../data', train=True, download=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

train_images = dataset.data
train_labels = dataset.targets

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(2)
   tensorflow.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

seed = 0
reset_random_seeds(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)
rng1 = np.random.RandomState(seed)

train_images1 = train_images.reshape((train_images.shape[0], -1, 1))
perm = rng1.permutation(train_images1.shape[1])
train_images1 = train_images1[:, perm]
train_images1 = train_images1.reshape(-1, 784)


reset_random_seeds(seed)
tensorflow.random.set_seed(seed)
np.random.seed(seed)
rng2 = np.random.RandomState(seed)

train_images2 = train_images.reshape((train_images.shape[0], -1, 1))
perm = rng2.permutation(train_images2.shape[1])
train_images2 = train_images2[:, perm]
train_images2 = train_images2.reshape(-1, 784)



data = './data/Dataset3'

with open(data, 'r+b') as handle:
    X_train = torch.Tensor(pickle.load(handle))
    y_train = torch.Tensor(pickle.load(handle))
    X_test = torch.Tensor(pickle.load(handle))
    y_test = torch.Tensor(pickle.load(handle))

X_train = torch.Tensor(np.concatenate([X_train, train_images1, train_images2]))
y_train = torch.Tensor(np.concatenate([y_train, train_labels, train_labels]))

X_train = X_train.type(torch.LongTensor)
y_train = y_train.type(torch.LongTensor)
X_test = X_test.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

X_train = X_train.reshape(-1,28,28)
X_test = X_test.reshape(-1,28,28)
    
def load_train(flatten=True):
    from torchvision import datasets, transforms

    x = X_train.float() / 255.
    y = y_train

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y

def load_test(flatten=True):
    from torchvision import datasets, transforms

    x = X_test.float() / 255.
    y = y_test

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_train(flatten=False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    flatten = True if config.model == 'fc' else False

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_test(flatten=False)
    
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
