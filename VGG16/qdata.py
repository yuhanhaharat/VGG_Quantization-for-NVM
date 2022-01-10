import os
import zipfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from PIL import Image  
import PIL  
import numpy as np
from cifar10_models.quantization import quantization

class qCIFAR10(Dataset):
    def __init__(self, cifar10, N=4, transform=None):
        self.data = cifar10.data
        self.targets = cifar10.targets
        self.transform = transform
        self.N = N

    def __getitem__(self, index):
        #Get labels for a single image
        single_image_target = self.targets[index]
        #Get single image
        single_image = self.data[index,:,:,:]
        #Quantization of single image
        test = quantization(self.N)
        qsingle_image_res = test.quantizeimage(single_image)
        if self.transform is not None:
            qimage_as_tensor = self.transform(qsingle_image_res)
        return (qimage_as_tensor, single_image_target)

    def __len__(self):
        return len(self.targets)
