from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
from centralized import Net, eval, train
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net: Net, trainloader: DataLoader, validloader: DataLoader, cfg: DictConfig) -> None:
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.cfg = cfg
        self.device = DEVICE
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        self.net.load_state_dict(state_dict=state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters=parameters)
        train(self.net, self.trainloader, parameters=self.cfg.parameters, device=self.device, verbose=False)
        return self.get_parameters(config=config), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters=parameters)
        metrics = eval(self.net, loader=self.validloader, device=self.device)
        return float(metrics['loss']), len(self.validloader), metrics
