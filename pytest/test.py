from typing import Any


class a():
    def __init__(v) -> None:
        v.a1 = 1

    def __call__(self,aaa):
        print(self.a1)
        return aaa
    
# class c(a):
#     def __init__(self) -> None:
#         a.__init__(self)
#         print(self.a1)
#     def cf(self):
#         return self.a1

class b(a):
    def __init__(self) -> None:
        aa = a()
        a.__init__(self)
        print(self.a1)
        self.a1 = 2
        

aaa = b()

# a = [4]
# b = []
# c = [2,5]
# print(a+b+c)
# # from torch import tensor
# # import torch
# # x = tensor([[1],[2],[3]])
# # print(torch.transpose(x,1,0)@x)
# # print(x**2)

# xk = {
#     "value": [[1,2,3],[2,3,5]],
#     "SEN": [2,3],
# }
# print(xk)
# print(xk["value"])
# print(xk["SEN"])
