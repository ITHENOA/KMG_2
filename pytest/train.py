import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from somfnn import SOMFNN
from dataset import load_dataset


def main():
    X, Y = load_dataset("pen")
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y)
    print(f"x shape: {X.shape} \nY shape: {Y.shape}")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)

    # Create dataset and dataloader
    dataset = TensorDataset(Xtr, Ytr)
    train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    dataset = TensorDataset(Xte, Yte)
    test_dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    # Initialize network
    net = SOMFNN(in_features=X.shape[-1], hidden_features=[3, 2], out_features=10)
    net.set_options(num_epochs=100, learning_rate=0.01, criterion="CrossEntropy", optimizer="Adam", training_plot=False)
    net.trainnet(train_dataloader)
    net.testnet(test_dataloader)


if __name__ == "__main__":
    main()
