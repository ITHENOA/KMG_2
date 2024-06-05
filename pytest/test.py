class b:
    def __init__(self):
        
        self.cc = 3
    def access():
        ch = super().aa
        print("100")
    
    @property
    def dd(self):
        return self.cc + 10

class c:
    def __init__(self):
        self.gg = 56

class a(b,c):
    def __init__(self):
        b.__init__(self)
        c.__init__(self)
        self.aa = 1
        self.bb = 2
        print(self.cc)
        print(self.dd)
        self.cc = 5
        print(self.dd)
        print(self.gg)


aaa = a()

# a = [4]
# b = []
# c = [2,5]
# print(a+b+c)
# # from torch import tensor
# # import torch
# # x = tensor([[1],[2],[3]])
# # print(torch.transpose(x,1,0)@x)
# # print(x**2)