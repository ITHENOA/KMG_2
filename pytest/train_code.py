import torch
import torchmetrics
from torch.utils.data import DataLoader
# from time import time
from torchmetrics import Accuracy

from my_dataset import MyDataset, split
from SOLAYER import Solayer

def main():
    ### Model
    class SOMFNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.so1 = Solayer(16,10)

        def forward(self, x):
            x = self.so1(x)
            return x

    ### Parameters
    device = 'cpu'
    batch_size = 32
    loader_workers = 2
    # schaduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    schaduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy(task="multiclass", num_classes=10)
    n_epochs = 10
    
    ### load dataset
    dataset = MyDataset('pen', dtype=torch.float32)
    train_dataset, test_dataset, val_dataset = split(dataset, test_ratio=0.2, val_ratio=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)

    ### Training Loop
    model = SOMFNN().to(device)
    n_iters = int(len(train_dataset)/n_epochs)
    for epoch in range(n_epochs):
        ## train
        model.train()
        tr_loss, tr_preds, tr_labels = 0., [], []
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            # forward
            yh = model(X)
            # calculate loss
            loss = criterion(yh, Y.long()) # classification
            # backward
            loss.backward()
            # update
            optimizer.param_groups[0]['params'] = list(model.parameters()) # handle new parameters because of self-organizing layers
            optimizer.step()
            optimizer.zero_grad()
            # save loss
            tr_loss += loss.item()
            tr_preds.extend(yh.argmax(1).cpu())
            tr_labels.extend(Y.cpu())
        
        tr_loss /= len(train_loader)
        tr_acc = metric(torch.tensor(tr_preds), torch.tensor(tr_labels)) * 100
        # metric.plot()
        
        ## Validation
        if len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_preds, val_labels = 0., [], []
                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                    yh_val = model(X_val)
                    val_loss += criterion(yh_val, Y_val.long()).item()
                    val_preds.extend(yh_val.argmax(1).cpu())
                    val_labels.extend(Y_val.cpu())
                val_loss /= len(val_loader)
                val_acc = metric(torch.tensor(val_preds), torch.tensor(val_labels)) * 100
            
            schaduler.step(val_loss)
            print(f"epoch[{epoch+1}/{n_epochs}], tr_loss[{tr_loss:.5f}], tr_acc[{tr_acc:.3f}], val_loss[{val_loss:.5f}], val_acc[{val_acc:.3f}]")
        else:
            schaduler.step()
            print(f"epoch[{epoch+1}/{n_epochs}], tr_loss[{tr_loss:.5f}], tr_acc[{tr_acc:.3f}]")


if __name__ == "__main__":
    main()