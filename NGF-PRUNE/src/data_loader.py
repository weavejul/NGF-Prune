from typing import Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .constants import (
    DATA_DIR, 
    MNIST_NORMALIZE_MEAN, 
    MNIST_NORMALIZE_STD,
    FASHION_MNIST_NORMALIZE_MEAN, 
    FASHION_MNIST_NORMALIZE_STD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEST_BATCH_SIZE
)

def get_mnist_loaders(batch_size: int = DEFAULT_BATCH_SIZE, test_batch_size: int = DEFAULT_TEST_BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """MNIST Data Loading"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_NORMALIZE_MEAN, MNIST_NORMALIZE_STD)
    ])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def get_fashion_mnist_loaders(batch_size: int = DEFAULT_BATCH_SIZE, test_batch_size: int = DEFAULT_TEST_BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """FashionMNIST Data Loading"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_NORMALIZE_MEAN, FASHION_MNIST_NORMALIZE_STD)
    ])
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader