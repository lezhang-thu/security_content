import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    dir_name = os.path.join('..', 'data')
    mnist_train = torchvision.datasets.FashionMNIST(root=dir_name,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=dir_name,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(
        mnist_train,
        batch_size,
        shuffle=True,
    ), data.DataLoader(
        mnist_test,
        batch_size,
        shuffle=False,
    ))


if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print("{} {}\n{} {}".format(X.shape, X.dtype, y.shape, y.dtype))
        break
