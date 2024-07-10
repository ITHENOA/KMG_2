import pandas as pd

def load_dataset(name):

    if name == "pen":
        data = pd.read_excel("pytest\data\pen_class10.xlsx")
        data = data.to_numpy()
        return data[:, :-1], data[:, -1]