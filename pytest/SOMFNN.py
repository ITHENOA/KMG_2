import torch
from torch import nn
import torch.optim as optim
from torch.functional import F 
import numpy as np

# -------------------------------RULE----------------------------------
class Rule:
    def __init__(rul, NO=1, xk=None, SEN_xk=None):
        rul.NO = NO
        rul.P = xk
        rul.Cc = xk
        rul.CX = 1
        rul.CS = 1

    
    
    def rule_density(rul,x,pn,taun):
        SEN_x = rul.square_euclidean_distance(x,pn)
        return torch.exp(-(SEN_x/taun)**2)

    def update(R):
        pass

# ------------------------------LAYER-----------------------------------
class Layer:
    def __init__(lay, NO=None, M=None, W=None):
        # Rule.__init__(net)
        lay.NO = NO # layer number
        lay.M = M # number of inputs
        lay.W = W # number of outputs
        lay.N = 0 # number of rules
        lay.g_mu = torch.zeros(M,1) # Global Mean
        lay.g_X = 0 #?  # Global Mean of Squared Eugliducian Norm
        lay.rules_info = []

    def __call__(lay, xk):
        SENx = SEN(xk)
        
        if lay.N == 0:
            lay.init_rule(xk, SENx)
        else:
            lay.add_rule()

    def init_rule(lay, xk, SENx):
        Rule(1)
        pass

    def add_rule(lay):
        lay.rules_info.append(Rule(lay.N))
        lay.N += 1

    
        
# --------------------------------NET---------------------------------
class SOMFNN(nn.Module):
    def __init__(net, in_features=3, hidden_features=[], out_features=1):
        super(SOMFNN, net).__init__()
        # nn.Module.__init__(net)
        # Layer.__init__(net)
        if isinstance(in_features, int): in_features = [in_features] 
        if isinstance(out_features, int): out_features = [out_features] 
        net.neurons = in_features + hidden_features + out_features # [M1, W1=M2, W2=M3, ..., Wend-1=Mend, Wend]
        net.n_layers = len(net.neurons) - 1
        net.layer_info = []
        net.fc = nn.ModuleList()
        for l in range(net.n_layers):
            net.fc.append(nn.Linear(net.neurons[l], net.neurons[l+1]))
            net.layer_info.append(Layer(l, net.neurons[l], net.neurons[l+1]))
        net.criterion = None
        net.optimizer = None
        net.n_epoch = None
        # net.device = net.get_device()
        net.device = get_device()
        net.to(net.device)

    def forward(net, x):
        # x.shape : (MB,feature)
        for l in range(net.n_layer):
            layer = net.layer_info[l](x)
            x = F.sigmoid(net.fc[l](x))
            x = net.lambda_func(l, x, lamb)
        return x

    def lambda_func(net, l, yn, lamb):

        return yn
        
    def options(net, n_epoch=10, lr=0.01, criterion='MSE', optimizer='SGD'):
        # Set the criterion
        if criterion == 'MSE':
            net.criterion = nn.MSELoss()
        elif criterion == 'BCE':
            net.criterion = nn.BCELoss()
        elif criterion == 'CrossEntropy':
            net.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion}")
        
        # Set the optimizer
        if optimizer == 'SGD':
            net.optimizer = optim.SGD(net.parameters(), lr=lr)
        elif optimizer == 'Adam':
            net.optimizer = optim.Adam(net.parameters(), lr=lr)
        elif optimizer == 'RMSprop':
            net.optimizer = optim.RMSprop(net.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")
        
        net.n_epoch = n_epoch


    def trainnet(net, X, Y):
        X, Y = X.to(net.device), Y.to(net.device)
        for epoch in range(net.n_epoch):
            net.train()
            # Forward pass
            outputs = net(X)
            loss = net.criterion(outputs, Y)

            # Backward pass and optimization
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{net.n_epoch}], Loss: {loss.item():.4f}")

        return net # trained net


    def testnet(net, X, Y):
        X, Y = X.to(net.device), Y.to(net.device)
        net.eval()
        with torch.no_grad():
            outputs = net(X)
            loss = net.criterion(outputs, Y)
            print(f'Test Loss: {loss.item():.4f}')

# -----------------------------------------------------------------
def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device
    
# square_euclidean_distance
def SEN(x):
    return torch.sqrt(torch.sum((x)**2))