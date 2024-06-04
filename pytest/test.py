import torch
from torch import tensor
a = tensor([1,2,3])
b = tensor([])
c = tensor([6])
d = torch.cat((a,b,c)).int()
print(a)
print(b)
print(c)
print(d)