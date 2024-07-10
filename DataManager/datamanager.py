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

def EMNIST():
    EMNIST_train = datasets.EMNIST(root='../dataset', split='balanced', train=True , download=False, transform=transforms.ToTensor())
    EMNIST_test  = datasets.EMNIST(root='../dataset', split='balanced', train=False, download=False, transform=transforms.ToTensor())
    return EMNIST_train, EMNIST_test

def FashionMNIST():
    FashionMNIST_train = datasets.FashionMNIST(root='../dataset', train=True , download=False, transform=transforms.ToTensor())
    FashionMNIST_test  = datasets.FashionMNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
    return FashionMNIST_train, FashionMNIST_test

if __name__ == '__main__':
    MNIST_train = datasets.MNIST(root='../dataset', train=True, download=False, transform=transforms.ToTensor())
    MNIST_test = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())

