import torch
from torch import nn
import torch.optim as optim

class SOMFNN(nn.Module):
    def __init__(self, in_features=3, hidden_features=[], out_features=1):
        super(SOMFNN, self).__init__()
        # self.neurons = [in_features, hidden_features, out_features]
        # self.n_layer = len(self.neurons) + 2
        self.fc = []
        for l in range(1, self.n_layer + 1):
            self.fc.append(nn.Linear(self.neurons[l-1], self.neurons[l]))
        self.criterion = None
        self.optimizer = None
        self.n_epoch = None
        self.device = self.get_device()
        self.to(self.device)


    def forward(self, x):
        for layer in range(1,self.n_layer):
            x = self.fc[layer](x)
            x = self.lambda_func(self, layer, x)
        return x
    

    def options(self, n_epoch=10, lr=0.01, criterion='MSE', optimizer='SGD'):
        # Set the criterion
        if criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'BCE':
            self.criterion = nn.BCELoss()
        elif criterion == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion}")
        
        # Set the optimizer
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")
        
        self.n_epoch = n_epoch


    def train(self, X, Y):
        X, Y = X.to(self.device), Y.to(self.device)
        for epoch in range(self.n_epoch):
            self.train()
            # Forward pass
            outputs = self(X)
            loss = self.criterion(outputs, Y)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{self.n_epoch}], Loss: {loss.item():.4f}")


    def test(self, X, Y):
        X, Y = X.to(self.device), Y.to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            loss = self.criterion(outputs, Y)
            print(f'Test Loss: {loss.item():.4f}')


    @staticmethod
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
