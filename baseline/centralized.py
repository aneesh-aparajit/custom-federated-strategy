import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def train(net: Net, loader: DataLoader, parameters: DictConfig, device: torch.device, verbose: bool = True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=parameters.learning_rate)
    net.train()

    for epoch in range(parameters.epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for X, y in tqdm(loader, total=len(loader), desc='train'):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
        epoch_loss  /= len(loader.dataset)
        epoch_acc  = correct / total

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


@torch.no_grad()
def eval(net: Net, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    net.eval()

    with torch.no_grad():
        correct, total, epoch_loss = 0, 0, 0.0
        for X, y in tqdm(loader, total=len(loader), desc='valid'):
            X, y = X.to(device), y.to(device)
            outputs = net.forward(X)
            loss = criterion(outputs, y)

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
    loss  /= len(loader.dataset)
    accuracy  = correct / total

    return {"loss": loss, "accuracy": accuracy}
