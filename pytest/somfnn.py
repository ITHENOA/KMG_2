import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from layer import Layer
from utils import get_device


class SOMFNN(nn.Module):
    """
    Self-Organizing Multilayer Fuzzy Neural Network class.
    """

    def __init__(
        self, in_features: int = 3, hidden_features: list = None, out_features: int = 1):
        super(SOMFNN, self).__init__()

        # check inputs
        if hidden_features is None:
            hidden_features = []
        if isinstance(in_features, int):
            in_features = [in_features]
        elif not isinstance(in_features, list):
            raise TypeError("'in_features' must be an integer or a list of integers.")
        if isinstance(out_features, int):
            out_features = [out_features]
        elif not isinstance(out_features, list):
            raise TypeError("'out_features' must be an integer or a list of integers.")

        # Set parameters
        self.neurons = in_features + hidden_features + out_features
        self.num_layers = len(self.neurons) - 1
        
        # Create the fully connected layers and layer information objects
        self.fc_layers = nn.ModuleList()
        self.layers_info = []
        for i in range(self.num_layers):
            self.fc_layers.append(nn.Linear(self.neurons[i], self.neurons[i + 1]))
            self.layers_info.append(Layer(i + 1, self.neurons[i], self.neurons[i + 1]))

        # Set options
        self.criterion = None
        self.optimizer = None
        self.num_epochs = None
        self.device = get_device()
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.
        """
        # Loop over all layers
        for i in range(self.num_layers):

            X_detached = X.detach()
            with torch.no_grad():
                # Compute lambda functions for the current layer
                lambdas = self.layers_info[i](X_detached)
            # Update the structure of the current pytorch fc layer
            self.add_neurons(i)
            # Compute the output of the current layer
            X = torch.sigmoid(self.fc_layers[i](X))
            # Compute the next input for the network
            X = self.apply_rule_strength(i, X, lambdas)

        return X

    def add_neurons(self, layer_index: int) -> None:
        """
        Update the structure of the fully connected layers.
        """
        layer = self.layers_info[layer_index]
        # new in_features and out_features
        in_features = layer.in_features
        new_out_features = layer.num_rules * layer.out_features
        old_out_features = self.fc_layers[layer_index].out_features
        
        if new_out_features != old_out_features: # requaires new neurons
            # create new fully connected layer
            new_fc = nn.Linear(in_features, new_out_features)
            # copy weights and biases
            new_fc.weight.data[:old_out_features] = self.fc_layers[layer_index].weight.data.clone()
            new_fc.bias.data[:old_out_features] = self.fc_layers[layer_index].bias.data.clone()
            # update self.layers[layer_index]
            self.fc_layers[layer_index] = new_fc

    def apply_rule_strength(self, layer_index: int, X: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute the lambda function for the given layer.
        """     
        # Extract number of outputs and rules from the layer info
        n_outputs = self.layers_info[layer_index].out_features
        n_samples, n_rules = lambdas.shape
        
        # create lambda-eye matricx -> shape(n_samples, n_outputs, n_outputs * n_rules)
        lambda_eye = torch.zeros(n_samples, n_outputs, n_outputs * n_rules)
        for m, sample in enumerate(lambdas):
            for n, rul in enumerate(sample):
                lambda_eye[m, :, n*n_outputs:(n+1)*n_outputs] = rul * torch.eye(n_outputs)
        
        # Apply the lambda matrices to the input tensor
        # lambda_eye.shape: (n_samples, n_outputs, n_outputs * n_rules)
        # X.shape: (n_samples, n_inputs) --unsqueeze--> (n_samples, 1, n_inputs) --mT--> (n_samples, n_inputs, 1)
        # result.shape: (n_samples, n_outputs, 1) -> (n_samples, n_outputs)
        X = torch.einsum('sou,sul->so', lambda_eye, X.unsqueeze(1).mT) # s:sample, o:output, u:rule*output
        # X = torch.matmul(lambda_eye, X.unsqueeze(1).mT).squeeze(2) # another way
        
        return X

    def set_options(self, num_epochs: int = 10, 
                    learning_rate: float = 0.01, 
                    criterion: str = "MSE", 
                    optimizer: str = "SGD") -> None:
        """
        Set the training options for the network.
        """
        if criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "BCE":
            self.criterion = nn.BCELoss()
        elif criterion == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion}")

        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")

        self.num_epochs = num_epochs

    def trainnet(net, dataloader: DataLoader) -> None:
        """
        Train the network with the given data.
        """
        if dataloader.dataset.tensors[0].shape[1] != net.neurons[0]:
            raise ValueError("Input dimension mismatch")
        if dataloader.dataset.tensors[1].shape[1] != net.neurons[-1]:
            raise ValueError("Output dimension mismatch")

        # Initialize the learning curve list
        loss_list = []

        for epoch in range(net.num_epochs):
            net.train()
            for X, Y in dataloader:
                X, Y = X.to(net.device), Y.to(net.device)
                net.optimizer.zero_grad()
                outputs = net(X)
                loss = net.criterion(outputs, Y)
                loss.backward()
                net.optimizer.step()

                # Append the loss to the learning curve list
                loss_list.append(loss.item())

            # Plot the learning curve
            plt.plot(loss_list)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning Curve')
            plt.show()

            # Print the loss for the current epoch
            print(f"Epoch {epoch+1}/{net.num_epochs}, Loss: {loss.item():.4f}")

    def testnet(self, dataloader: DataLoader) -> None:
        """
        Test the network with the given data.
        """
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f"Test Loss: {average_loss:.4f}")
