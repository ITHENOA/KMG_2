import torch

a = torch.tensor(1., requires_grad=True)
b = torch.tensor([],requires_grad=False)
b = a * 2
print(a)