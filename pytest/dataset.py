import pandas as pd
from scipy.io import loadmat
import numpy as np
from torch import tensor, cat, empty

def load_dataset(name):

    if name == "pen":
        data = pd.read_excel("pytest\\data\\pen_class10.xlsx")
        data = data.to_numpy()
        return data[:, :-1], data[:, -1]
    
    if name == "sp500":
        data = pd.read_excel("pytest\\data\\SP500_reg.xlsx")
        data = data.to_numpy()
        data = data[:,5] # close price
        X = tensor([])
        Y = tensor([])
        for t in range(5,len(data)-1):
            x = tensor([data[t-4], data[t-3], data[t-2], data[t-1], data[t]]).unsqueeze(0)
            X = cat((X,x), dim=0)
            Y = cat((Y, tensor(data[t+1]).unsqueeze(0)))
        return X, Y
    
    if name == "muscle_sim_8cls":
        data = loadmat("pytest\\data\\matlab.mat")["ans"]
        X = data[:,:-1]
        Y = data[:,-1]
        return tensor(X), tensor(Y)
