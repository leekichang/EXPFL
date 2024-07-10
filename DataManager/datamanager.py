import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def MNIST():
    MNIST_train  = datasets.MNIST(root='../dataset', train=True, download=False, transform=transforms.ToTensor())
    MNIST_test   = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
    return MNIST_train, MNIST_test

if __name__ == '__main__':
    MNIST_train = datasets.MNIST(root='../dataset', train=True, download=False, transform=transforms.ToTensor())
    MNIST_test = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())

