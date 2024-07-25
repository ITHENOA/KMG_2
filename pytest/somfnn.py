import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from time import time
from torchmetrics import Accuracy
# F.linear(x)

from layer import Layer
# from utils import get_device
from Solayer import Solayer


class SOMFNN(nn.Module):
    """
    Self-Organizing Multilayer Fuzzy Neural Network class.
    """

    def __init__(self, in_features: int = 3, hidden_features: list = [], out_features: int = 1, device="cpu"):
        super(SOMFNN, self).__init__()

        if device == "cuda" and not torch.cuda.is_available():
            print("cuda is not avalable.")
            device = "cpu"
        self.device = device
        print(f"using {self.device} device ...")
        self.to(self.device)
        
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
        # self.weights = nn.Parameter(tor)
        
        # Set options
        self.loss_fn = None
        self.optimizer = None
        self.num_epochs = None
        
        # Create the fully connected layers and layer information objects
        # self.fc_layers = nn.ModuleList()
        # self.layers_info = []
        # for i in range(self.num_layers):
        #     self.fc_layers.append(nn.Linear(self.neurons[i], self.neurons[i + 1]))
        #     self.layers_info.append(Layer(i + 1, self.neurons[i], self.neurons[i + 1], device=self.device))

        # self.fc1 = nn.Linear(16,10)
        # self.solay1 = Layer(1,16,10, device=self.device)
        # self.fc2 = nn.Linear(30,30)
        # self.solay2 = Layer(1,30,30, device=self.device)
        # self.fc3 = nn.Linear(30,10)
        # self.solay3 = Layer(1,30,10, device=self.device)
        
        self.solay1 = Solayer(16,10)
        

    # -----------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.

        X.shape = (Batch, in_features)
        """
        # # Loop over all layers
        # for l in range(self.num_layers):
        #     with torch.no_grad():
        #         # Compute lambda functions for the current layer
        #         lambdas = self.layers_info[l](X) # (samples, rules)
        #         # Update the structure of the current pytorch fc layer
        #         self.add_neurons(l)
        #     # Compute the output of the current layer
        #     X = self.fc_layers[l](X) # X.shape = (Batch, Rule * out_features)
        #     X = F.sigmoid(X)
        #     # Compute the next input for the network
        #     X = self.apply_rule_strength(l, X, lambdas) # X.shape = (Batch, out_features)

        # # Layer: 1
        # with torch.no_grad(): lamb = self.solay1(X)
        # self.fc1 = self.add(self.solay1, self.fc1)
        # # w = torch.cat()
        # X = self.fc1(X)
        # X = F.sigmoid(X)
        # X = self.apply_lamb(self.solay1, X, lamb)
        # # # Layer: 2
        # # with torch.no_grad(): lamb = self.solay2(X)
        # # self.fc2 = self.add(self.solay2, self.fc2)
        # # X = F.sigmoid(self.fc2(X))
        # # X = self.apply_lamb(self.solay2, X, lamb)
        # # # Layer: 3
        # # with torch.no_grad(): lamb = self.solay3(X)
        # # self.fc3 = self.add(self.solay3, self.fc3)
        # # X = (self.fc3(X))
        # # X = self.apply_lamb(self.solay3, X, lamb)
        
        X = self.solay1(X)

        return X

    # @staticmethod
    # def add(solay, fc) -> None:
    #     in_features = solay.in_features
    #     new_out_features = solay.out_features_per_rule * solay.n_rules
    #     old_out_features = fc.out_features
    #     if new_out_features != old_out_features: # requaires new neurons
    #         new_fc = nn.Linear(in_features, new_out_features)
    #         new_fc.weight.data[:old_out_features] = fc.weight.data.clone()
    #         new_fc.bias.data[:old_out_features] = fc.bias.data.clone()
    #         return new_fc.to(fc.weight.device.type)
    #     return fc
        
        
    # @staticmethod    
    # def apply_lamb(solay: int, X: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    #     n_outputs = solay.out_features_per_rule
    #     n_samples, n_rules = lambdas.shape
    #     # return X.reshape([n_samples, n_rules, n_outputs]).transpose(1,2).sum(2)
    #     return torch.einsum("lRS,ROS->lOS",
    #         lambdas.transpose(1,0).reshape([1, n_rules, n_samples]),
    #         X.transpose(1,0).reshape([n_rules, n_outputs, n_samples])
        # ).transpose(1,0).reshape(n_samples, n_outputs)
            
            
    # -----------------------------------------------------------------------
    def add_neurons(self, layer_index: int) -> None:
        """
        Update the structure of the fully connected layers.
        """
        layer = self.layers_info[layer_index]
        # new in_features and out_features
        in_features = layer.in_features
        new_out_features = layer.out_features_per_rule * layer.n_rules
        # new_out_features = layer.num_rules * layer.out_features
        old_out_features = self.fc_layers[layer_index].out_features
        # old_out_features = self.fc_layers[layer_index].out_features
        
        if new_out_features != old_out_features: # requaires new neurons
            # create new fully connected layer
            new_fc = nn.Linear(in_features, new_out_features)
            
            # copy previous weights and biases
            old_weights = self.fc_layers[layer_index].weight.data.clone()
            old_biases = self.fc_layers[layer_index].bias.data.clone()
            
            # randimize weights initialization
            new_fc.weight.data[:old_out_features] = old_weights
            new_fc.bias.data[:old_out_features] = old_biases
            
            # mean weights initialization
            if self.init_weights_type == "mean":
                n_added_rules = int((new_out_features - old_out_features) / layer.out_features_per_rule)
                init_weights = torch.reshape(old_weights, [layer.out_features_per_rule, old_weights.shape[1], layer.n_rules - n_added_rules]).mean(2)
                init_biases = torch.reshape(old_biases, [layer.out_features_per_rule, layer.n_rules - n_added_rules]).mean(1)
                new_fc.weight.data[old_out_features:] = init_weights.repeat(n_added_rules, 1)
                new_fc.bias.data[old_out_features:] = init_biases.repeat(n_added_rules)

            if self.init_weights_type == "in_paper":
                new_fc.weight.data[old_out_features:] = torch.randint(0, 2, [new_out_features - old_out_features, in_features]) / (in_features+1)
                new_fc.bias.data[old_out_features:] = torch.randint(0, 2, [new_out_features - old_out_features]) / (in_features+1)

            # update self.layers[layer_index]
            self.fc_layers[layer_index] = new_fc

    # -----------------------------------------------------------------------
    def apply_rule_strength(self, layer_index: int, X: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute the lambda function for the given layer.

        X.shape = (batch, Rule * out_features)
        lambdas.shape = (batch, Rule)

        return.shape = (batch, out_features)
        """     
        # Extract number of outputs and rules from the layer info
        # n_outputs = self.layers_info[layer_index].out_features_per_rule
        n_outputs = self.neurons[layer_index + 1]
        # n_outputs = layer_index
        n_samples, n_rules = lambdas.shape
        
        ### method 1 ### [647s for rep=10000, batch=128]
        # s = time()
        # for _ in range(10000):
        #     # create lambda-eye matricx -> shape(n_samples, n_outputs, n_outputs * n_rules)
        #     lambda_eye = torch.zeros(n_samples, n_outputs, n_outputs * n_rules)
        #     for m, sample in enumerate(lambdas):
        #         for n, rul in enumerate(sample):
        #             lambda_eye[m, :, n*n_outputs:(n+1)*n_outputs] = rul * torch.eye(n_outputs)
        #     # Apply the lambda matrices to the input tensor
        #     # lambda_eye.shape: (n_samples, n_outputs, n_outputs * n_rules)
        #     # X.shape: (n_samples, n_inputs) --unsqueeze--> (n_samples, 1, n_inputs) --mT--> (n_samples, n_inputs, 1)
        #     # result.shape: (n_samples, n_outputs, 1) -> (n_samples, n_outputs)
        #     X1 = torch.einsum('sou,sul->so', lambda_eye, X.unsqueeze(1).mT) # s:sample, o:output, u:rule*output
        # print("1",time()-s)

        ### method 2 ### [586s for rep=1000, batch=128]
        # s = time()
        # for _ in range(10000):
        #     # create lambda-eye matricx -> shape(n_samples, n_outputs, n_outputs * n_rules)
        #     lambda_eye = torch.zeros(n_samples, n_outputs, n_outputs * n_rules)
        #     for m, sample in enumerate(lambdas):
        #         for n, rul in enumerate(sample):
        #             lambda_eye[m, :, n*n_outputs:(n+1)*n_outputs] = rul * torch.eye(n_outputs)
        #     # Apply the lambda matrices to the input tensor
        #     # lambda_eye.shape: (n_samples, n_outputs, n_outputs * n_rules)
        #     # X.shape: (n_samples, n_inputs) --unsqueeze--> (n_samples, 1, n_inputs) --mT--> (n_samples, n_inputs, 1)
        #     # result.shape: (n_samples, n_outputs, 1) -> (n_samples, n_outputs)
        #     X2 = torch.matmul(lambda_eye, X.unsqueeze(1).mT).squeeze(2) # another way
        # print("2",time()-s)

        ### method 3 ### [106s for rep=1000, batch=128]
        # s = time()
        # for _ in range(10000):
        #     X3 = torch.zeros([n_samples, n_outputs])
        #     for i, (lamb_B, X_B) in enumerate(zip(lambdas, X)):
        #         X3[i,:] = (X_B.reshape((n_rules, n_outputs)).t() * lamb_B.unsqueeze(0)).sum(1)
        # print("3",time()-s)

        ### method 4 ### [2s for rep=1000, batch=128]
        # s = time()
        # for _ in range(10000):
        #     X4 = torch.einsum("lrs,ros->los",
        #         lambdas.mT.unsqueeze(0),
        #         X.mT.reshape([n_rules, n_outputs, n_samples])
        #     ).mT.squeeze(0)
        # print("4",time()-s)

        ### method 5 ### [2s for rep=1000, batch=128]
        # l:1, R:n_rule, O:n_out, S:n_sample
        # return torch.einsum("lRS,ROS->lOS",
        #     lambdas.mT.reshape([1, n_rules, n_samples]),
        #     X.mT.reshape([n_rules, n_outputs, n_samples])
        # ).mT.reshape(n_samples, n_outputs)
        return torch.einsum("lRS,ROS->lOS",
            lambdas.transpose(1,0).reshape([1, n_rules, n_samples]),
            X.transpose(1,0).reshape([n_rules, n_outputs, n_samples])
        ).transpose(1,0).reshape(n_samples, n_outputs)

    # -----------------------------------------------------------------------
    def set_options(self, num_epochs: int = 10, 
                    learning_rate: float = 0.01, 
                    criterion: str = "MSE", 
                    optimizer: str = "SGD",
                    init_weights_type: str = None,
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
        self.init_weights_type = init_weights_type

    # -----------------------------------------------------------------------
    def trainnet(self, train_loader: DataLoader, verbose: bool = True, val_loader: DataLoader = None) -> None:
        """
        Train the network with the given data.
        """
        self = self.to(self.device)
        
        if train_loader.dataset.tensors[0].shape[1] != self.neurons[0]:
            raise ValueError("Input dimension mismatch")
        # if dataloader.dataset.tensors[1].ndim == 1:
        #     if dataloader.dataset.tensors[1].shape[1] != net.neurons[-1]:
        #         raise ValueError("Output dimension mismatch")

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        if self.training_plot: 
            plt.ion() # Enable interactive mode
            fig, axs = plt.subplots(2, figsize=(10, 8))

        
        for epoch in range(self.num_epochs):
            

            # Training
            self.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for X, Y in train_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                if self.loss_fn_name == "CE": Y = Y.long()

                # Compute prediction error
                Yhat = self(X)
                loss = self.loss_fn(Yhat, Y)

                # Backpropagation
                loss.backward()
                # self.optimizer = optim.Adam(self.parameters(), lr=1)
                self.optimizer.param_groups[0]['params'] = list(self.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # print(Yhat.argmax(1) == Y)

                train_loss += loss.item()
                if self.loss_fn_name == "CE":
                    train_preds.extend(Yhat.argmax(1).cpu().numpy())
                elif self.loss_fn_name == "MSE":
                    train_preds.extend(Yhat.detach().cpu().numpy())
                train_labels.extend(Y.cpu().numpy())

            train_loss /= len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            #train_accuracy = mean_squared_error(train_labels, train_preds)
            # train_f1 = f1_score(train_labels, train_preds, average='weighted')

            # Validation
            if val_loader:
                self.eval()
                val_loss = 0.0
                val_preds, val_labels = [], []

                with torch.no_grad():
                    for X, Y in val_loader:
                        Yhat = self(X)
                        loss = self.loss_fn(Yhat, Y)

                        val_loss += loss.item()
                        if self.loss_fn_name == "CE":
                            val_preds.extend(torch.argmax(Yhat, dim=1).cpu().numpy())
                        elif self.loss_fn_name == "MSE":
                            val_preds.extend(Yhat.cpu().numpy())
                        val_labels.extend(Y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_accuracy = Accuracy()
                # val_accuracy = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # Append the losses and accuracies
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            if val_loader:
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            # Plot the learning curve
            if self.training_plot:
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
            # for i in range(net.num_layers):
            #     rules.append(net.layers_info[i].n_rules)

            # Print the loss for the current epoch
            if verbose:
                if val_loader:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: [{train_loss:.2f}], Train Acc: [{train_accuracy*100:.2f}], rules: {rules}")

        print("Done")
        # Plot the final learning curve
        if self.training_plot:    
            plt.ioff() # Disable interactive mode after training is done
            plt.show() # Show the final plot
        
    # -----------------------------------------------------------------------
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

def SOlayer(in_features, out_features):
    fc = nn.Linear(in_features,out_features)
    so = Layer(1,in_features,out_features)