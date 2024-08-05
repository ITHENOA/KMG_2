
import torch
import torch.nn.functional as F

from CORELAYER import Corelayer
# from somfnn import SOMFNN

class Solayer(torch.nn.Module):
    def __init__(self, in_features, out_features, init_weights_type=None):
        super(Solayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)
        self.core = Corelayer(in_features, out_features)
        # self.activation = getattr(F, activation)
        self.init_weights_type = init_weights_type
        self.first_call = True
    
    def forward(self, X, activation=None):
        
        if self.first_call:
            self.device = X.device.type
            self.first_call = False
        
        # Compute lambda functions for the current layer
        # with torch.no_grad(): lambdas = self.core(X) # (samples, rules)
        lambdas = self.core(X.detach().cpu()).to(self.device) # (samples, rules)
        # Update the structure of the current pytorch fc layer
        self.add_neurons()
        # Compute the output of the current layer
        X = self.fc(X) # X.shape = (Batch, Rule * out_features)
        X = getattr(F, activation)(X) if activation is not None else X # dont use for last layer
        # Compute the next input for the network
        X = self.apply_rule_strength(X, lambdas) # X.shape = (Batch, out_features)
        
        return X
        
    # -----------------------------------------------------------------------
    def add_neurons(self) -> None:
        """
        Update the structure of the fully connected layers.
        """
        # new in_features and out_features
        in_features = self.core.in_features
        new_out_features = self.core.out_features_per_rule * self.core.n_rules
        old_out_features = self.fc.out_features
        
        if new_out_features != old_out_features: # requaires new neurons
            # create new fully connected layer
            new_fc = torch.nn.Linear(in_features, new_out_features, device=self.device)
            
            # # copy previous weights and biases
            # old_weights = self.fc.weight.data.clone()
            # old_biases = self.fc.bias.data.clone()
            # # randimize weights initialization
            # new_fc.weight.data[:old_out_features] = old_weights
            # new_fc.bias.data[:old_out_features] = old_biases

            with torch.no_grad():
                new_fc.weight[:old_out_features] = self.fc.weight
                new_fc.bias[:old_out_features] = self.fc.bias
            
            # mean weights initialization
            if self.init_weights_type == "mean":
                n_added_rules = int((new_out_features - old_out_features) / self.core.out_features_per_rule)
                init_weights = torch.reshape(old_weights, [self.core.out_features_per_rule, old_weights.shape[1], self.core.n_rules - n_added_rules]).mean(2)
                init_biases = torch.reshape(old_biases, [self.core.out_features_per_rule, self.core.n_rules - n_added_rules]).mean(1)
                new_fc.weight.data[old_out_features:] = init_weights.repeat(n_added_rules, 1)
                new_fc.bias.data[old_out_features:] = init_biases.repeat(n_added_rules)

            if self.init_weights_type == "in_paper":
                new_fc.weight.data[old_out_features:] = torch.randint(0, 2, [new_out_features - old_out_features, in_features]) / (in_features+1)
                new_fc.bias.data[old_out_features:] = torch.randint(0, 2, [new_out_features - old_out_features]) / (in_features+1)

            # update self.layers[layer_index]
            self.fc = new_fc

    # -----------------------------------------------------------------------
    def apply_rule_strength(self, X: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute the lambda function for the given layer.

        X.shape = (batch, Rule * out_features)
        lambdas.shape = (batch, Rule)

        return.shape = (batch, out_features)
        """     
        # Extract number of outputs and rules from the layer info
        n_outputs = self.core.out_features_per_rule
        # n_outputs = self.neurons[layer_index + 1]
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
        return torch.einsum("lRS,ROS->lOS",
            lambdas.mT.reshape([1, n_rules, n_samples]),
            X.mT.reshape([n_rules, n_outputs, n_samples])
        ).mT.reshape(n_samples, n_outputs)
        # return torch.einsum("lRS,ROS->lOS",
        #     lambdas.transpose(1,0).reshape([1, n_rules, n_samples]),
        #     X.transpose(1,0).reshape([n_rules, n_outputs, n_samples])
        # ).transpose(1,0).reshape(n_samples, n_outputs)