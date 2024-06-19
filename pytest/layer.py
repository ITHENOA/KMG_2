import torch
from torch import tensor, cat

from utils import squared_euclidean_norm, get_device, delta


class Layer:
    """
    A class representing a layer in the neural network.
    """

    def __init__(self, layer_number: int = None, in_features: int = None,
                 out_features: int = None) -> None:
        self.layer_number = layer_number  # layer number
        self.in_features = in_features  # number of inputs
        self.out_features = out_features  # number of outputs
        self.num_rules = 0  # number of rules
        self.global_mean = torch.zeros(in_features)  # Global Mean, shape([num_inputs])
        self.global_sen_mean = tensor(0.)  # Global Mean of Squared Euclidean Norm
        self.prototypes = torch.empty(0)  # (rules, features)
        self.centroids = torch.empty(0) # (rules, features) mean of samples in clusters
        self.sen_centroids = torch.empty(0)  # (rules, 1) squared Euclidean norm of samples
        self.support = []  # (rules, 1) number of samples in clusters
        self.num_seen_samples = 0
        # self.device = get_device()
        # self.to(self.device)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        SEN = squared_euclidean_norm(X)
        self.update_global_parameters(X, SEN)

        sample_rules = []  # clusters NO of each sample

        for x, sen in zip(X, SEN):
            # sen becomes 1D matrix
            sen = sen.unsqueeze(0)
            # x becomes 2D matrix
            x = x.unsqueeze(0)
            if self.num_rules == 0:
                self.initialize_rule(x, sen)
                # update "rule index of each sample" vector
                sample_rules.append(0)
            else:
                logit, rule_index = self.rule_condition(x)
                if logit:
                    self.initialize_rule(x, sen)
                    sample_rules.append(self.num_rules - 1)
                else:
                    self.update_rule(rule_index, x, sen)
                    sample_rules.append(rule_index)

        return self.compute_lambdas(X)  # (samples, rules)

    def compute_lambdas(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute and normalize densities for each sample.
        """
        densities = self.local_density(X) # (samples, rules)
        density_sum = torch.sum(densities, dim=1, keepdim=True) # sum over rules
        lambdas = densities / density_sum # (samples, rule)
        return lambdas

    def rule_condition(self, x: torch.Tensor) -> tuple:
        densities = self.local_density(x) # (samples, rules)
        value, index = torch.max(densities, dim=1) # max over rules
        return value[0] < delta, index[0].item()

    def local_density(self, X: torch.Tensor, kernel: str = "RBF") -> torch.Tensor:
        # stau.shape: (rules) --unsqueeze--> (1, rules)
        stau = abs(
            self.global_sen_mean - squared_euclidean_norm(self.global_mean) +
            self.sen_centroids - squared_euclidean_norm(self.centroids)
        ).unsqueeze(0) / 2 

        # X(sample, features)
        # prototypes(rules, features) --unsqueeze--> (rules, 1, features)
        diff = X - self.prototypes.unsqueeze(1) # (rules, samples, features)
        sen_diff = (diff * diff).sum(2).mT # (rules, samples) --mT--> (samples, rules)
        if kernel == "RBF":
            densities = (- sen_diff / stau).exp()
        else:
            raise ValueError("Invalid kernel type")

        return densities # (samples, rules)

    def update_global_parameters(self, X: torch.Tensor,
                                 SEN: torch.Tensor) -> None:
        """
        Update global parameters using a batch of samples.
        """
        self.num_seen_samples += len(SEN)
        self.global_mean += (torch.mean(X, dim=0) - self.global_mean) / \
                            self.num_seen_samples
        self.global_sen_mean += (torch.mean(SEN) - self.global_sen_mean) / self.num_seen_samples

    def initialize_rule(self, x: torch.Tensor, sen_x: torch.Tensor) -> None:
        """
        Initialize a new rule.
        """
        self.num_rules += 1
        self.prototypes = cat((self.prototypes, x), dim=0) # prototypes.ndim == x.ndim == 2
        self.centroids = cat((self.centroids, x)) # centroids.ndim == x.ndim == 2
        self.sen_centroids = cat((self.sen_centroids, sen_x), dim=0) 
        self.support.append(1)

    def update_rule(self, rule_index: int, x: torch.Tensor,
                    sen_x: torch.Tensor) -> None:
        """
        Update an existing rule.
        """
        if sen_x.ndim == 1: sen_x = sen_x.squeeze(0)
        
        self.support[rule_index] += 1
        self.centroids[rule_index] += (
            x.squeeze(0) - self.centroids[rule_index]
        ) / self.support[rule_index]
        self.sen_centroids[rule_index] += (
            sen_x - self.sen_centroids[rule_index]
        ) / self.support[rule_index]
