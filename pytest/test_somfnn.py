from SOMFNN import SOMFNN
import torch

# Generate random data
X = torch.randn(100, 3)
Y = torch.randn(100, 1)

# network
net = SOMFNN(input_feature=3, hidden_neorun=[], output_feature=1)
net.options(
    n_epoch=100, 
    lr=0.01, 
    criterion='MSE', 
    optimizer='Adam'
    )
net.train(X, Y)
net.test(X, Y)
