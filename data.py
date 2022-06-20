import os
import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, Omniglot


class MyDataset(Dataset):
    def __init__(self, names):
        self.names = names
        self.transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img = Image.open(name).convert('RGB')
        return self.transform(img), -1


def mnist_data_loader(mnist_folder='./data/mnist_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = MNIST(mnist_folder, train=True, download=False, transform=transform)
    test_set = MNIST(mnist_folder, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader


def fashion_mnist_data_loader(folder='./data/fashion_mnist_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = FashionMNIST(folder, train=True, download=False, transform=transform)
    test_set = FashionMNIST(folder, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader


def cifar10_data_loader(folder='./data/cifar10_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = CIFAR10(folder, train=True, download=False, transform=transform)
    test_set = CIFAR10(folder, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader


def cifar100_data_loader(folder='./data/cifar100_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = CIFAR100(folder, train=True, download=False, transform=transform)
    test_set = CIFAR100(folder, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader


def lsun_data_loader(folder='./data/lsun_data/', batch_size=64):
    # label -1
    img_names = [folder + name for name in os.listdir(folder)]
    data_set = MyDataset(img_names)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return data_loader


def svhn_data_loader(folder='./data/svhn_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = SVHN(folder, split='train', download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)

    val_set = SVHN(folder, split='test', download=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=12)

    extra_set = SVHN(folder, split='extra', download=False, transform=transform)
    extra_loader = DataLoader(extra_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, val_loader, extra_loader


def omniglot_data_loader(folder='./data/omniglot_data', batch_size=64):
    # ground truth label
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    data_set = Omniglot(folder, download=False, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return data_loader


def tinyImageNet_data_loader(folder='./data/tinyImageNet_data/', batch_size=64):
    # label -1
    img_names = [folder + name for name in os.listdir(folder)]
    data_set = MyDataset(img_names)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return data_loader


if __name__ == '__main__':
    # train_loader, val_loader = mnist_data_loader(batch_size=256)  # checked
    # train_loader, val_loader = fashion_mnist_data_loader(batch_size=256)  # checked
    # train_loader = omniglot_data_loader(batch_size=256)  # checked
    # train_loader, val_loader = cifar10_data_loader(batch_size=256)  # checked
    # train_loader, val_loader = cifar100_data_loader(batch_size=256)  # checked
    # train_loader, val_loader, extra_loader = svhn_data_loader(batch_size=256)  # checked
    train_loader = tinyImageNet_data_loader(batch_size=256)
    for x, y in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, x.max(), x.min(), y)