import torch

a = torch.rand(1,20).mean(1)
b = torch.rand(1,5)
c = torch.mean(a,b[0,0])
d = torch.mean(a,b)
print(c)
print(d)

for name, param in self.named_parameters():
    if param.requires_grad:
        print(f"Name: {name}, Shape: {param.shape}")