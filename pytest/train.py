import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from somfnn import SOMFNN


def main():
    # Generate random data
    X = torch.randn(100, 4)
    Y = torch.randn(100, 1)

    # Create dataset and dataloader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    torch.autograd.set_detect_anomaly(True)
    # Initialize network
    net = SOMFNN(in_features=4, hidden_features=[3, 2], out_features=1)
    net.set_options(num_epochs=100, learning_rate=0.01, criterion="MSE", optimizer="Adam")
    net.trainnet(dataloader)
    net.testnet(dataloader)


if __name__ == "__main__":
    main()
