import torch
import pandas as pd
from scipy.io import loadmat
import numpy as np
from torch import tensor, cat, empty
from torch.utils.data import Dataset, random_split


class MyDataset(Dataset):
    def __init__(self, name, dtype=torch.float32):
        
        if name == "pen":
            data = pd.read_excel("pytest\\data\\pen_class10.xlsx")
            data = data.to_numpy()
            data = torch.from_numpy(data).type(dtype)
            self.X, self.Y = data[:, :-1], data[:, -1]
    
        if name == "sp500":
            data = pd.read_excel("pytest\\data\\SP500_reg.xlsx")
            data = tensor(data, dtype=dtype)
            data = data[:,5] # close price
            X = tensor([])
            Y = tensor([])
            for t in range(5,len(data)-1):
                x = tensor([data[t-4], data[t-3], data[t-2], data[t-1], data[t]]).unsqueeze(0)
                self.X = cat((X,x), dim=0)
                self.Y = cat((Y, tensor(data[t+1]).unsqueeze(0)))
        
        if name == "muscle_sim_8cls":
            data = loadmat("pytest\\data\\matlab.mat")["ans"]
            data = tensor(data, dtype=dtype)
            self.X, self.Y = data[:,:-1], data[:,[-1]]
        
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
    
def split(dataset, test_ratio, val_ratio=0):
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    return train_dataset, test_dataset, val_dataset