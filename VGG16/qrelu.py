import torch
import torch.nn as nn
import numpy as np

class qrelu(nn.Module):
    def __init__(self, N=4, min=0, max=1):
        super(qrelu, self).__init__()
        self.N = N
        self.min = min
        self.max = max

    def forward(self, x):
        level = 2**self.N
        step=(self.max-self.min)/(level-1)
        xnew = x.detach().numpy()
        xnew[xnew < 0] = 0
        I = np.around(((xnew-self.min)/step))     
        I[I == level] = level-1
        I[I < 0] = 0
        res = self.min+I*step
        res = torch.from_numpy(res)  
        return res