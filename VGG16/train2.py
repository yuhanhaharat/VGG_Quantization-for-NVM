import torch
import torchvision
import torchvision.transforms as transforms

from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from data import CIFAR10Data
from module import CIFAR10Module
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np

args = edict(d = {"data_dir": "/data/huy/cifar10",
            "download_weights": 0,
            "test_phase": 1,
            "dev": 0,
            "logger": "tensorboard",
            "classifier": "vgg16_bn",
            "pretrained": 1,
            "precision": 32,
            "batch_size":1,
            "max_epochs":100,
            "num_workers":0,
            "gpu_id":"3",
            "learning_rate":1e-2,
            "weight_decay":1e-2})

my_model = vgg16_bn(pretrained=True)
my_model.eval() # for evaluation
data = CIFAR10Data(args)
dataloader = data.test_dataloader()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct = 0
total = 0
i = 0
with torch.no_grad():
    for x in dataloader:
        images, labels = x
        outputs = my_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i = i+1
        if i % 500 == 0:   
            print('Testing Progress Images [%d/10000]' %(i + 1))

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))