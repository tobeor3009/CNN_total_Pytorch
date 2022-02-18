from numpy import identity
from torch import nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TransformerEncoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x
