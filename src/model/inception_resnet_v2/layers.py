import torch
from torch import nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# Assume Channel First
class ConcatBlock(nn.Module):
    def __init__(self, layer_list, dim=1):
        super().__init__()
        self.layer_list = layer_list
        self.dim = dim

    def forward(self, x):
        tensor_list = [layer(x) for layer in self.layer_list]
        return torch.cat(tensor_list, self.dim)


class TransformerEncoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x
