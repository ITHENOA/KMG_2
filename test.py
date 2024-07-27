import torch
import torch.nn as nn

class cls1(nn.Module):
    def __init__(self, a, b, device):
        super(cls1, self).__init__()
        self.a = a
        self.b = b
        self.device = device
        self.to(device)
        self.cls2 = cls2(self)

class cls2(nn.Module):
    def __init__(self, parent_cls1):
        super(cls2, self).__init__()
        self.device = parent_cls1.device
        # Now you can use self.device in this class
        0
        
cls1(1,2,"cpu")