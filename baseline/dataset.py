import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


def load_datasets(num_clients: int, batch_size: int, seed: int = 42):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = CIFAR10(root="../dataset/", train=True, download=True, transform=transform)
    testset  = CIFAR10(root="../dataset/", train=False, download=True, transform=transform)

    partition_size = len(trainset) // num_clients
    lengths = [partition_size]*num_clients
    datasets = random_split(trainset, lengths=lengths, generator=torch.Generator().manual_seed(seed))

    trainloaders, validloaders = [], []
    for ds in datasets:
        valid_len = len(ds) // 10 # 10% validation
        train_len  = len(ds) - valid_len
        lengths = [train_len, valid_len]
        ds_train, ds_valid = random_split(ds, lengths=lengths, generator=torch.Generator().manual_seed(seed))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        validloaders.append(DataLoader(ds_valid, batch_size=batch_size, shuffle=False))
    
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, validloaders, testloader

