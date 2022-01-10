import collections
import numpy as np
import torch

class quantization:
    def __init__(self, N):
        self.N = N
            
    def quantize(self,example,minval,maxval):    
        level = 2**self.N
        step=(maxval-minval)/(level-1)
        I = np.around(((example.numpy()-minval)/step))     
        I[I == level] = level-1
        I[I < 0] = 0
        qexample = minval+I*step
        return qexample
    
    def quantizeimage(self,image):
        minval = 0
        maxval = 256
        level = 2**self.N
        step=(maxval-minval)/(level-1)
        qimage = np.ones_like(image)
        #Iterate RGB channels, quantize and combine them again
        for x in range(3):
            I = np.around(((image[:,:,x]-minval)/step))     
            I[I == level] = level-1
            I[I < 0] = 0
            qimage_ch = minval+I*step
            qimage[:,:,x] = qimage_ch
        return qimage
            
    def quantizeValue(self, params):
        res = collections.OrderedDict()
        for n, example in params.items():
            print(n)
            substring = "num_batches_tracked"
            if substring in n:
                res[n] = example
            else:
                maxval = torch.max(example).item()
                minval = torch.min(example).item()
                qexample = self.quantize(example,minval,maxval)
                res[n] = torch.from_numpy(qexample)    
        return res