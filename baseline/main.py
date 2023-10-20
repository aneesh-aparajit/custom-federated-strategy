import time

import hydra
import torch
import flwr as fl
from centralized import Net, eval, train
from client import FlowerClient
from dataset import load_datasets
from omegaconf import DictConfig, OmegaConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # load the datasets
    trainloaders, validloaders, testloader = load_datasets(num_clients=cfg.num_clients, batch_size=cfg.parameters.batch_size, seed=cfg.seed)
    
    # # Do a round of centralized training
    # print("\n\t\t============= CENTRALIZED TRAINING =============")
    # trainloader, validloader = trainloaders[0], validloaders[0]
    # net = Net().to(DEVICE)
    # for epoch in range(cfg.parameters.epochs):
    #     train(net=net, loader=trainloader, parameters=cfg.parameters, device=DEVICE, verbose=True)
    #     metrics = eval(net=net, loader=validloader, device=DEVICE)
    #     print(f"Epoch {epoch+1}: validation loss {metrics['loss']}, accuracy: {metrics['accuracy']}")

    def client_fn(cid: str, cfg: DictConfig) -> FlowerClient:
        net = Net().to(device=DEVICE)
        trainloader, validloader = trainloaders[int(cid)], validloaders[int(cid)]
        return FlowerClient(net=net, trainloader=trainloader, validloader=validloader, cfg=cfg)
    
    # Startegy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5, 
        min_fit_clients=10, 
        min_evaluate_clients=5, 
        min_available_clients=10
    )

    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources
    )

    return


if __name__ == '__main__':
    main()
