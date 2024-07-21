import torch
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from torchviz import make_dot
import time


from somfnn import SOMFNN
from dataset import load_dataset


def main():
    X, Y = load_dataset("pen")
    # X = StandardScaler.fit_transform(X) # normalize (mean=0, std=1)
    # X = MinMaxScaler.fit_transform(X, ) # normalize [0,1]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X = (X - X.min(0).values) / (X.max(0).values - X.min(0).values)
    # Y = (Y - Y.min(0).values) / (Y.max(0).values - Y.min(0).values)
    print(f"x shape: {X.shape} \nY shape: {Y.shape}")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)
    # Xtr, Xval, Ytr, Yval = train_test_split(Xtr, Ytr, test_size=0.1)

    batch_size = 10
    # Create dataset and dataloader
    dataset = TensorDataset(Xtr, Ytr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = TensorDataset(Xte, Yte)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # dataset = TensorDataset(Xval, Yval)
    # val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize network
    model = SOMFNN(in_features=X.shape[-1], hidden_features=[], out_features=10)
    model.set_options(
        num_epochs=50, 
        learning_rate=.1, 
        criterion="CE", # MSE | BCE | CE
        optimizer="SGD",  # SGD | Adam | RMSprop
        training_plot=False,
        init_weights_type=None # None(pytorch default) | in_paper | mean
    )
    
    model.trainnet(train_dataloader, val_loader=None, verbose=1)
    model.testnet(test_dataloader)

    ## export onnx
    # torch.onnx.export(model, Xte, "pytest\\model.onnx")
    # torch.onnx.dynamo_export(model, Xte).save("model.onnx")
    # torch.onnx.export(model, Xte, "model.onnx", opset_version=11,
    #               custom_opsets={"CustomOp": 1}, 
    #               export_params=True)
    # traced_model = torch.jit.trace(model, Xte)
    # torch.onnx.export(traced_model, Xte, "saved_models\\simple_model.onnx", opset_version=11, verbose=True)


    ## export torchviz graph
    # make_dot(net(X), params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")
    # graph = make_dot(net(X).mean(), params=dict(net.named_parameters()), show_attrs=True, show_saved=True)
    # graph.render("backward_graph", format="png")
    # graph

    # netron.app



if __name__ == "__main__":
    main()
