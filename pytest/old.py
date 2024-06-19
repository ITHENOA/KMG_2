import torch
from torch import nn, tensor, cat
import torch.optim as optim
from torch.functional import F

# import numpy as np
# from itertools import product

global k, delta
k = 1
delta = torch.exp(tensor([-2]))


# ------------------------------LAYER-----------------------------------
class Layer:
    def __init__(lay, NO=None, M=None, W=None):
        # Rule.__init__(net)
        lay.NO = NO  # layer number
        lay.M = M  # number of inputs
        lay.W = W  # number of outputs
        lay.N = 0  # number of rules
        lay.gmean = torch.zeros(1, M)  # Global Mean
        lay.SENgmean = tensor([0])  # Global Mean of Squared Euclidian Norm
        # lay.RULES = []
        lay.prototypes = tensor([])  # (rules,features)
        # lay.stau = tensor([]) #(rules,1)
        lay.c = tensor([])  # (rules,features) mean of samples in clusters
        lay.SENc = tensor([])  # (rules,1) squared Euclidian norm of samples in clusters
        lay.support = []  # (rules,1) number of samples in clusters
        # lay.sample_rule = tensor([])
        lay.n_seen_sample = 0

    def __call__(lay, xbatch):
        SENbatch = SEN(xbatch)
        lay.update_global_pars(xbatch, SENbatch)  # update with xbatch or x ???
        # sample_rules = torch.zeros_like(SENbatch, dtype=torch.int8) # clusters NO of each sample
        sample_rules = []  # clusters NO of each sample
        # lamb = tensor([])

        # sample by sample
        for i, (x, SENx) in enumerate(zip(xbatch, SENbatch)):
            SENx = SENx.unsqueeze(0)
            x = x.unsqueeze(0)
            # lay.update_global_pars(x, SENx) # update with xbatch or x ???
            if lay.N == 0:
                lay.init_rule(x, SENx)
                # lamb = cat([lamb, 1])
                # sample_rules[i] = 0 # clusters NO of each sample
                sample_rules.append(0)  # clusters NO of each sample
            else:
                logit, rule_star = lay.rule_condition(x)
                if logit:
                    lay.init_rule(x, SENx)
                    # sample_rules[i] = lay.N-1
                    sample_rules.append(lay.N - 1)
                else:
                    lay.update_rule(rule_star, x, SENx)
                    # sample_rules[i] = rule_star
                    sample_rules.append(rule_star)

        return lay.get_lambda(xbatch)

    def get_lambda(lay, X):
        # Compute local densities for each sample and rule
        denses = lay.local_dens(X)  # shape(rule, MB)

        # Compute sum of rule densities for each sample (across all rules)
        dens_xi = torch.sum(denses, dim=0, keepdim=True)  # shape(1, MB)

        # Normalize densities for each sample by the sum of densities
        lambs = denses / dens_xi  # shape(rule, MB)

        return lambs  # shape(rule, MB)

    def rule_condition(lay, x):
        denses = lay.local_dens(x)
        value, idx = torch.max(denses, 0)
        return value < delta, idx

    def local_dens(lay, X, kernel="RBF"):
        # X : (1,#features) or (MB,#features)
        stau = abs(lay.SENgmean - SEN(lay.gmean) + lay.SENc - SEN(lay.c)) / 2  # tau**2
        dens = torch.zeros(len(stau), X.shape[0])
        for i, x in enumerate(X):
            if kernel == "RBF":
                dens[:, i] = torch.exp(-SEN(x - lay.prototypes) / stau)
            else:
                raise ("invalid kernel type")

        return dens  # (rule,MB)

    def update_global_pars(lay, xbatch, SENbatch):
        # # for sample
        # lay.gmean = lay.gmean + (x - lay.gmean)/k
        # lay.SENgmean = lay.SENgmean + (SENx - lay.SENgmean)/k
        # lay.n_seen_sample += 1

        # for batch of samples
        lay.n_seen_sample += len(SENbatch)
        lay.gmean = (
            lay.gmean + (torch.mean(xbatch, dim=0) - lay.gmean) / lay.n_seen_sample
        )  # k == lay.n_seen_sample  ?
        lay.SENgmean = (
            lay.SENgmean + (torch.mean(SENbatch) - lay.SENgmean) / lay.n_seen_sample
        )

    def init_rule(lay, x, SENx):
        lay.N += 1
        lay.prototypes = cat([lay.prototypes, x], dim=0)
        lay.c = cat([lay.c, x])
        lay.SENc = cat([lay.SENc, SENx], dim=0)
        lay.support.append(1)  # = cat([lay.support, tensor([1], dtype=torch.int64)])
        # lay.RULES.append(Rule(lay.N, x, SENx))

    def update_rule(lay, rule_star, x, SENx):
        lay.support[rule_star] = +1
        lay.c[rule_star] = (
            lay.c[rule_star] + (x - lay.c[rule_star]) / lay.support[rule_star]
        )
        lay.SENc[rule_star] = (
            lay.SENc[rule_star] + (SENx - lay.SENc[rule_star]) / lay.support[rule_star]
        )


# --------------------------------NET---------------------------------
class SOMFNN(nn.Module):
    def __init__(net, in_features=3, hidden_features=[], out_features=1):
        super(SOMFNN, net).__init__()
        # nn.Module.__init__(net)
        # Layer.__init__(net)
        if isinstance(in_features, int):
            in_features = [in_features]
        if isinstance(out_features, int):
            out_features = [out_features]
        net.neurons = (
            in_features + hidden_features + out_features
        )  # [M1, W1=M2, W2=M3, ..., Wend-1=Mend, Wend]
        net.n_layers = len(net.neurons) - 1
        # net.pars = parameter_box()
        net.LAYERS = []
        net.fc = nn.ModuleList()
        for l in range(net.n_layers):
            net.fc.append(nn.Linear(net.neurons[l], net.neurons[l + 1]))
            net.LAYERS.append(Layer(l + 1, net.neurons[l], net.neurons[l + 1]))
        net.criterion = None
        net.optimizer = None
        net.n_epoch = None
        # net.device = net.get_device()
        net.device = get_device()
        net.to(net.device)

    def forward(net, x):
        # x.shape : (MB,feature)
        for l in range(net.n_layers):
            lambs = net.LAYERS[l](x)  # also net.LAYER will be update
            yn = F.sigmoid(net.fc[l](x))
            x = net.lambda_func(l, yn, lambs)

        return x, net

    def lambda_func(net, l, yn, lambs):
        W = net.LAYERS[l].W  # number of outputs
        R = len(lambs)  # number of rules
        y = torch.zeros(W, W * R)
        for lamb in lambs:
            y = cat([yn, torch.eye(W) * lamb], dim=1)

        return y

    def options(net, n_epoch=10, lr=0.01, criterion="MSE", optimizer="SGD"):
        # Set the criterion
        if criterion == "MSE":
            net.criterion = nn.MSELoss()
        elif criterion == "BCE":
            net.criterion = nn.BCELoss()
        elif criterion == "CrossEntropy":
            net.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion}")

        # Set the optimizer
        if optimizer == "SGD":
            net.optimizer = optim.SGD(net.parameters(), lr=lr)
        elif optimizer == "Adam":
            net.optimizer = optim.Adam(net.parameters(), lr=lr)
        elif optimizer == "RMSprop":
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

        return net  # trained net

    def testnet(net, X, Y):
        X, Y = X.to(net.device), Y.to(net.device)
        net.eval()
        with torch.no_grad():
            outputs = net(X)
            loss = net.criterion(outputs, Y)
            print(f"Test Loss: {loss.item():.4f}")


# -----------------------------------------------------------------
def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device


# square_euclidean_distance
def SEN(x):
    return torch.sqrt(torch.sum((x) ** 2, dim=1))
    # return torch.sum((x)**2, dim=0) # 0 for vector


def rule_density(rul, x, pn, taun):
    SEN_x = rul.square_euclidean_distance(x, pn)
    return torch.exp(-((SEN_x / taun) ** 2))
