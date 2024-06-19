import torch
from torch import nn, tensor, cat
import torch.optim as optim
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset

k: int = 1
delta = torch.exp(tensor([-2]))

def get_device() -> str:
    """
    Get the device to be used for computation.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device

def squared_euclidean_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared Euclidean norm of a tensor.
    """
    return torch.sum(x ** 2, dim=1)

class Layer:
    """
    A class representing a layer in the neural network.
    """

    def __init__(self, layer_number: int = None, num_inputs: int = None,
                 num_outputs: int = None) -> None:
        self.layer_number = layer_number  # layer number
        self.num_inputs = num_inputs  # number of inputs
        self.num_outputs = num_outputs  # number of outputs
        self.num_rules = 0  # number of rules
        self.global_mean = torch.zeros(1, num_inputs)  # Global Mean
        self.global_sen_mean = tensor([0])  # Global Mean of Squared Euclidean Norm
        self.prototypes = tensor([])  # (rules, features)
        self.centroids = tensor([])  # (rules, features) mean of samples in clusters
        self.sen_centroids = tensor([])  # (rules, 1) squared Euclidean norm of samples
        self.support = []  # (rules, 1) number of samples in clusters
        self.num_seen_samples = 0

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        sen_batch = squared_euclidean_norm(x_batch)
        self.update_global_parameters(x_batch, sen_batch)

        sample_rules = []  # clusters NO of each sample

        for x, sen_x in zip(x_batch, sen_batch):
            sen_x = sen_x.unsqueeze(0)
            x = x.unsqueeze(0)
            if self.num_rules == 0:
                self.initialize_rule(x, sen_x)
                sample_rules.append(0)
            else:
                logit, rule_index = self.rule_condition(x)
                if logit:
                    self.initialize_rule(x, sen_x)
                    sample_rules.append(self.num_rules - 1)
                else:
                    self.update_rule(rule_index, x, sen_x)
                    sample_rules.append(rule_index)

        return self.compute_lambdas(x_batch)

    def compute_lambdas(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute and normalize densities for each sample by the sum of densities.
        """
        densities = self.local_density(x_batch)
        density_sum = torch.sum(densities, dim=0, keepdim=True)
        lambdas = densities / density_sum
        return lambdas

    def rule_condition(self, x: torch.Tensor) -> tuple:
        densities = self.local_density(x)
        value, index = torch.max(densities, 0)
        return value < delta, index

    def local_density(self, x_batch: torch.Tensor, kernel: str = "RBF") -> torch.Tensor:
        """
        Calculate local densities for each sample and rule.
        """
        stau = abs(
            self.global_sen_mean - squared_euclidean_norm(self.global_mean) +
            self.sen_centroids - squared_euclidean_norm(self.centroids)
        ) / 2

        densities = torch.zeros(len(stau), x_batch.shape[0])
        for i, x in enumerate(x_batch):
            if kernel == "RBF":
                densities[:, i] = torch.exp(
                    -squared_euclidean_norm(x - self.prototypes) / stau
                )
            else:
                raise ValueError("Invalid kernel type")

        return densities

    def update_global_parameters(self, x_batch: torch.Tensor,
                                 sen_batch: torch.Tensor) -> None:
        """
        Update global parameters using a batch of samples.
        """
        self.num_seen_samples += len(sen_batch)
        self.global_mean += (torch.mean(x_batch, dim=0) - self.global_mean) / \
                            self.num_seen_samples
        self.global_sen_mean += (torch.mean(sen_batch) - self.global_sen_mean) / \
                                self.num_seen_samples

    def initialize_rule(self, x: torch.Tensor, sen_x: torch.Tensor) -> None:
        """
        Initialize a new rule.
        """
        self.num_rules += 1
        self.prototypes = cat([self.prototypes, x], dim=0)
        self.centroids = cat([self.centroids, x])
        self.sen_centroids = cat([self.sen_centroids, sen_x], dim=0)
        self.support.append(1)

    def update_rule(self, rule_index: int, x: torch.Tensor,
                    sen_x: torch.Tensor) -> None:
        """
        Update an existing rule.
        """
        self.support[rule_index] += 1
        self.centroids[rule_index] += (
            x - self.centroids[rule_index]
        ) / self.support[rule_index]
        self.sen_centroids[rule_index] += (
            sen_x - self.sen_centroids[rule_index]
        ) / self.support[rule_index]


class SOMFNN(nn.Module):
    """
    Self-Organizing Map-based Fuzzy Neural Network class.
    """

    def __init__(self, in_features: int = 3, hidden_features: list = None,
                 out_features: int = 1) -> None:
        super(SOMFNN, self).__init__()

        if hidden_features is None:
            hidden_features = []

        if isinstance(in_features, int):
            in_features = [in_features]
        if isinstance(out_features, int):
            out_features = [out_features]

        self.neurons = in_features + hidden_features + out_features
        self.num_layers = len(self.neurons) - 1
        self.layers = nn.ModuleList()
        self.layers_objects = []

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.neurons[i], self.neurons[i + 1]))
            self.layers_objects.append(
                Layer(i + 1, self.neurons[i], self.neurons[i + 1])
            )

        self.criterion = None
        self.optimizer = None
        self.num_epochs = None
        self.device = get_device()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        for i in range(self.num_layers):
            lambdas = self.layers_objects[i](x)
            y_n = torch.sigmoid(self.layers[i](x))
            x = self.lambda_function(i, y_n, lambdas)

        return x

    def lambda_function(self, layer_index: int, y_n: torch.Tensor,
                        lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute the lambda function for the given layer.
        """
        num_outputs = self.layers_objects[layer_index].num_outputs
        num_rules = lambdas.shape[0]
        y = torch.zeros(lambdas.shape[1], num_outputs * num_rules)
        for i, lamb in enumerate(lambdas):
            y[:, i * num_outputs: (i + 1) * num_outputs] = y_n * lamb.unsqueeze(1)
        return y

    def set_options(self, num_epochs: int = 10, learning_rate: float = 0.01,
                    criterion: str = "MSE", optimizer: str = "SGD") -> None:
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

    def train_network(self, dataloader: DataLoader) -> None:
        """
        Train the network with the given data.
        """
        for epoch in range(self.num_epochs):
            self.train()
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self(x_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item():.4f}")

    def test_network(self, dataloader: DataLoader) -> None:
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


# Test script
if __name__ == "__main__":
    # Generate random data
    X = torch.randn(100, 3)
    Y = torch.randn(100, 1)

    # Create dataset and dataloader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Initialize network
    net = SOMFNN(in_features=3, hidden_features=[3, 2], out_features=1)
    net.set_options(num_epochs=100, learning_rate=0.01, criterion="MSE", optimizer="Adam")
    net.train_network(dataloader)
    net.test_network(dataloader)
