import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score


from layer import Layer
from utils import get_device


class SOMFNN(nn.Module):
    """
    Self-Organizing Multilayer Fuzzy Neural Network class.
    """

    def __init__(
        self, in_features: int = 3, hidden_features: list = [], out_features: int = 1):
        super(SOMFNN, self).__init__()

        # check inputs
        # if hidden_features is None:
        #     hidden_features = []
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

        self.fc = nn.Linear(16,10)
        self.lay = Layer(1,16,10)

        # Set options
        self.loss_fn = None
        self.optimizer = None
        self.num_epochs = None
        self.device = get_device()
        self.to(self.device)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.
        """
        # Loop over all layers
        for l in range(self.num_layers):
            # X = X.detach()
            with torch.no_grad():
                # Compute lambda functions for the current layer
                lambdas = self.layers_info[l](X) # (samples, rules)
            # Update the structure of the current pytorch fc layer
            self.add_neurons(l)
            # Compute the output of the current layer
            X = F.sigmoid(self.fc_layers[l](X))
            # Compute the next input for the network
            X = self.apply_rule_strength(l, X, lambdas)

        # with torch.no_grad():
        #     lamb = self.lay(X)
        # self.add()
        # X = self.fc(X)
        # X = F.sigmoid(X)
        # X = self.apply_lamb(10, X, lamb)

        return X

    def add(self) -> None:
        """
        Update the structure of the fully connected layers.
        """
        layer = self.lay
        # new in_features and out_features
        in_features = layer.in_features
        new_out_features = layer.out_features_per_rule * layer.num_rules
        old_out_features = self.fc.out_features
        
        if new_out_features != old_out_features: # requaires new neurons
            # create new fully connected layer
            new_fc = nn.Linear(in_features, new_out_features)
            # copy weights and biases
            new_fc.weight.data[:old_out_features] = self.fc.weight.data.clone()
            new_fc.bias.data[:old_out_features] = self.fc.bias.data.clone()
            # update self.layers[layer_index]
            self.fc = new_fc


    def apply_lamb(self, n_outputs: int, X: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute the lambda function for the given layer.
        """     
        # Extract number of outputs and rules from the layer info
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
        # X = torch.einsum('sou,sul->so', lambda_eye, X.unsqueeze(1).mT) # s:sample, o:output, u:rule*output
        X = torch.matmul(lambda_eye, X.unsqueeze(1).mT).squeeze(2) # another way
        
        return X

    def add_neurons(self, layer_index: int) -> None:
        """
        Update the structure of the fully connected layers.
        """
        layer = self.layers_info[layer_index]
        # new in_features and out_features
        in_features = layer.in_features
        new_out_features = layer.out_features_per_rule * layer.num_rules
        # new_out_features = layer.num_rules * layer.out_features
        old_out_features = self.fc_layers[layer_index].out_features
        # old_out_features = self.fc_layers[layer_index].out_features
        
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
        n_outputs = self.layers_info[layer_index].out_features_per_rule
        # n_outputs = self.layers_info[layer_index].out_features
        # n_outputs = layer_index
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
        # X = torch.einsum('sou,sul->so', lambda_eye, X.unsqueeze(1).mT) # s:sample, o:output, u:rule*output
        X = torch.matmul(lambda_eye, X.unsqueeze(1).mT).squeeze(2) # another way
        
        return X


    def set_options(self, num_epochs: int = 10, 
                    learning_rate: float = 0.01, 
                    criterion: str = "MSE", 
                    optimizer: str = "SGD",
                    training_plot: bool = False,
                    validation_ratio: float = 0):
        """
        Set the training options for the network.
        """
        if criterion == "MSE":
            self.loss_fn = nn.MSELoss()
            self.loss_fn_name = "MSE"
        elif criterion == "BCE":
            self.loss_fn = nn.BCELoss()
            self.loss_fn_name = "BCE"
        elif criterion == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
            self.loss_fn_name = "CE"
        else:
            raise ValueError(f"Unsupported criterion type: {criterion}")

        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            # self.optimizer_name = "SGD"
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            # self.optimizer_name = "Adam"
        elif optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
            # self.optimizer_name = "RMSprop"
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")

        self.num_epochs = num_epochs
        self.training_plot = training_plot
        self.validation_ratio = validation_ratio


    def trainnet(net, train_loader: DataLoader, verbose: bool = True, val_loader: DataLoader = None) -> None:
        """
        Train the network with the given data.
        """
        if train_loader.dataset.tensors[0].shape[1] != net.neurons[0]:
            raise ValueError("Input dimension mismatch")
        # if dataloader.dataset.tensors[1].ndim == 1:
        #     if dataloader.dataset.tensors[1].shape[1] != net.neurons[-1]:
        #         raise ValueError("Output dimension mismatch")

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        if net.training_plot: 
            plt.ion() # Enable interactive mode
            fig, axs = plt.subplots(2, figsize=(10, 8))

        
        for epoch in range(net.num_epochs):

            # Training
            net.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for X, Y in train_loader:
                X, Y = X.to(net.device), Y.to(net.device)
                if net.loss_fn_name == "CE": Y = Y.long()

                # Compute prediction error
                Yhat = net(X)
                loss = net.loss_fn(Yhat, Y)

                # Backpropagation
                loss.backward()
                net.optimizer.step()
                net.optimizer.zero_grad()

                train_loss += loss.item()
                if net.loss_fn_name == "CE":
                    train_preds.extend(Yhat.argmax(1).cpu().numpy())
                elif net.loss_fn_name == "MSE":
                    train_preds.extend(Yhat.detach().cpu().numpy())
                train_labels.extend(Y.cpu().numpy())

            train_loss /= len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')

            # Validation
            if val_loader:
                net.eval()
                val_loss = 0.0
                val_preds, val_labels = [], []

                with torch.no_grad():
                    for X, Y in val_loader:
                        Yhat = net(X)
                        loss = net.loss_fn(Yhat, Y)

                        val_loss += loss.item()
                        if net.loss_fn_name == "CE":
                            val_preds.extend(torch.argmax(Yhat, dim=1).cpu().numpy())
                        elif net.loss_fn_name == "MSE":
                            val_preds.extend(Yhat.cpu().numpy())
                        val_labels.extend(Y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # Append the losses and accuracies
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            if val_loader:
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            # Plot the learning curve
            if net.training_plot:
                axs[0].clear()
                axs[0].plot(range(1, epoch+2), train_losses, label='Train Loss')
                axs[0].plot(range(1, epoch+2), val_losses, label='Val Loss')
                axs[0].legend()
                axs[0].set_title('Loss')
                
                axs[1].clear()
                axs[1].plot(range(1, epoch+2), train_accuracies, label='Train Accuracy')
                axs[1].plot(range(1, epoch+2), val_accuracies, label='Val Accuracy')
                axs[1].legend()
                axs[1].set_title('Accuracy')
                
                plt.pause(0.01)

            rules = []
            for i in range(net.num_layers):
                rules.append(net.layers_info[i].num_rules)

            # Print the loss for the current epoch
            if verbose:
                if val_loader:
                    print(f"Epoch {epoch+1}/{net.num_epochs}, Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{net.num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, rules: {rules}")

        print("Done")
        # Plot the final learning curve
        if net.training_plot:    
            plt.ioff() # Disable interactive mode after training is done
            plt.show() # Show the final plot
        

    def testnet(net, dataloader: DataLoader) -> None:
        """
        Test the network with the given data.
        """
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(net.device), y_batch.to(net.device)
                if net.loss_fn_name == "CE": y_batch = y_batch.long()
                outputs = net(x_batch)
                loss = net.loss_fn(outputs, y_batch)
                total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f"Test Loss: {average_loss:.4f}")
