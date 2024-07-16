import torch
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from torchviz import make_dot


from somfnn import SOMFNN
from dataset import load_dataset


def main():
    X, Y = load_dataset("pen")
    # X = StandardScaler.fit_transform(X) # normalize (mean=0, std=1)
    # X = MinMaxScaler.fit_transform(X, ) # normalize [0,1]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X = (X - X.min(0).values) / (X.max(0).values - X.min(0).values)
    print(f"x shape: {X.shape} \nY shape: {Y.shape}")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)
    # Xtr, Xval, Ytr, Yval = train_test_split(Xtr, Ytr, test_size=0.1)

    # Create dataset and dataloader
    dataset = TensorDataset(Xtr, Ytr)
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    dataset = TensorDataset(Xte, Yte)
    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    # dataset = TensorDataset(Xval, Yval)
    # val_dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    # Initialize network
    net = SOMFNN(in_features=X.shape[-1], hidden_features=[], out_features=10)
    net.set_options(num_epochs=20, 
                    learning_rate=1, 
                    criterion="CE", # 'MSE', 'BCE', 'CE'
                    optimizer="Adam",  # 'SGD', 'Adam', 'RMSprop'
                    training_plot=False,
                    init_weights_type=None)
    
    net.trainnet(train_dataloader, val_loader=None, verbose=1)
    net.testnet(test_dataloader)

    # Yh = net(X)
    # make_dot(Yh, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")
    # graph = make_dot(Yh.mean(), params=dict(net.named_parameters()), show_attrs=True, show_saved=True)
    # graph.render("backward_graph", format="png")
    # graph



if __name__ == "__main__":
    main()
