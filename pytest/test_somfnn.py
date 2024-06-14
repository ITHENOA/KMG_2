from SOMFNN import SOMFNN
import torch

# Generate random data
X = torch.randn(3, 20)
Y = torch.randn(1, 20)

# network
net = SOMFNN(in_features=3, hidden_features=[3, 2], out_features=1)
net.options(n_epoch=100, lr=0.01, criterion="MSE", optimizer="Adam")
net.trainnet(X, Y)
net.testnet(X, Y)
