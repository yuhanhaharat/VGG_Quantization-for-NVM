# VGG16 Quantization for Hardware Implementation

# Quantization in General

We need several information for quantization. Maxval, minval, input x and number of bits N we wish to quantize for. We need calcualte number of levels and step in order to calcuate what is quantized x (qx). The following code is doing this. example here is a tensor. Numpy array can do quantization for the whole array instead of just a number (no for loop is needed)

```python
    def quantize(self,example,minval,maxval):    
        level = 2**self.N
        step=(maxval-minval)/(level-1)
        #we convert tensor to numpy
        I = np.around(((example.numpy()-minval)/step))     
        I[I == level] = level-1
        I[I < 0] = 0
        qexample = minval+I*step
    return qexample
```

# Without Quantization (full precision)

We first load pretrained VGG16 model from https://github.com/huyvnphan/PyTorch_CIFAR10

We import the model like this

```python
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
my_model = vgg16_bn(pretrained=True)
my_model.eval() # for evaluation
```
We download the CIFAR10 data like this
```python
data = CIFAR10Data(args)
dataloader = data.test_dataloader()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Note that we pass args in above. Actually if we don't use terminal to pass the args like parse.args() as lots of program generlly do, we will need to use edict in python to generate the args. It will acts equivalently as you do it in terminal.

if you do this in terminal
```python
if __name__ == "__main__":
    #parser = ArgumentParser()
    # PROGRAM level args
    #parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    #parser.add_argument("--download_weights", type=int, default=1, choices=[0, 1])
    #parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
    #parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    #parser.add_argument(
    #    "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    #)
    # TRAINER args
    #parser.add_argument("--classifier", type=str, default="vgg16_bn")
    #parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1])
    #parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    #parser.add_argument("--batch_size", type=int, default=256)
    #parser.add_argument("--max_epochs", type=int, default=100)
    #parser.add_argument("--num_workers", type=int, default=8)
    #parser.add_argument("--gpu_id", type=str, default="3")
    #parser.add_argument("--learning_rate", type=float, default=1e-2)
    #parser.add_argument("--weight_decay", type=float, default=1e-2)
    #args = parser.parse_args()
    main(args)
```  
if you do this directly in main function
```python
from easydict import EasyDict as edict
if __name__ == "__main__":
    args = edict(d = {"data_dir": "/data/huy/cifar10",
            "download_weights": 0,
            "test_phase": 1,
            "dev": 0,
            "logger": "tensorboard",
            "classifier": "vgg16_bn",
            "pretrained": 1,
            "precision": 32,
            "batch_size":256,
            "max_epochs":100,
            "num_workers":8,
            "gpu_id":"3",
            "learning_rate":1e-2,
            "weight_decay":1e-2})

    print(args.download_weights)
    main(args)
```

We do the inference on this model like this, print results every 500 images.
```python
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
```

# Quantization of Weights

We first need to quantize the weights of a pretrianed VGG16 model. This is called post training quantization. The idea of this quantization is basically just modift the pretrained weights directly from state_dict to res and load res as a eventual model for inference. 

```python
def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        #print weights in state_dict of a specific layer.
        print(state_dict["features.0.bias"])
        N = 5
        #weight quantization part
        #create a class object
        test1 = quantization(N)
        #call a method for this object
        res = test1.quantizeValue(state_dict)
        #load weights or qweights
        model.load_state_dict(res)
        model.load_state_dict(state_dict)
    return model
```

Here we create a class called quantization and we use quantizaValue() for this purpose. Since state_dict is a dictionary that contain weights from all layers so we need pass this to the quantizeValue() for it.

```python
def quantizeValue(self, params):
        #Key of this dictionary will arranged based on inseration order
        res = collections.OrderedDict()
        #Iterate n (layer name) and example (tensor)
        for n, example in params.items():
            print(n)
            #No quantization for these layers
            substring = "num_batches_tracked"
            if substring in n:
                res[n] = example
            #Array based quantization for the rest of the layers
            else:
                maxval = torch.max(example).item()
                minval = torch.min(example).item()
                qexample = self.quantize(example,minval,maxval)
                #Add quantization result to the res[] dict
                res[n] = torch.from_numpy(qexample)    
        return res
```
# Quantization of Inputs

Now, we quantize the CIFAR10 image. Usually for creating a custom pytorch dataset class that can be used for dataloader, we write a class inheritanced from Dataset. 

Import some packages
```python
from torch.utils.data import Dataset, DataLoader
from cifar10_models.quantization import quantization
```

The class method has two functions. getitem() and len(). getitem return (samples and lables) as a tensor which can be directly used in dataloader.
This class do some processing per single image.
```python
class qCIFAR10(Dataset):
    def __init__(self, cifar10, N=4, transform=None):
        #Get original cifar10 data using .data and .targets
        self.data = cifar10.data
        self.targets = cifar10.targets
        #Transform method
        self.transform = transform
        #Quantization level
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
```
For quantize a image, we quantize R,G,B channel seperately [min:0,max:256] and combined them.
```python
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
```
After quantization, we use the results here:
```python
def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        #Origianl data
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=True)
        #Original data print (10000,32,32,3)
        a = dataset.data[1,:,:,:]
        b = dataset.targets[1]
        #Quantization data
        qdataset = qCIFAR10(dataset,N=8,transform=transform)
        #Cannot call qdataset like this and it will return the original dataset
        qdataset.data[1,:,:,:]
        dataloader = DataLoader(qdataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
```

# Quantization of Activiations
Simililarly, VGG16 use ReLU across all the layers so we want to quantize the ReLU. In pytorch, we could write a class inheritance from nn.module to create our own functions. The best tutorial for this is from:https://morioh.com/p/deaf2f23fbc6

```python
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
        #Do ReLU quantization here
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
```
Then, we will use this layer in the forward path. Since nn.sequential() combine all the layers so in the def forward() function we need get output from each layers to quantize it. Therefore, we iterate through the layers in self.features() and self.classifiers() using a for loop. In sum, the block of code use qrelu() instead of relu(). 

```python
def forward(self, x):
        for layer in self.features:  
            if isinstance(layer, nn.ReLU):
                maxval = torch.max(x).item()
                m = qrelu(3,0,maxval)
                x = m(x)
                #x = F.relu(x)
            else:
                x = layer(x)
        #x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        for layer in self.classifier:  
            if isinstance(layer, nn.ReLU):
                maxval = torch.max(x).item()
                m = qrelu(3,0,maxval)
                x = m(x)
                #x = F.relu(x)
            else:
                x = layer(x)
        return x
```
If we want to print the model summary and a certain layer information, we could use the following code.
```python
if(arch == "vgg16_bn"):
    print(model)
    print(model.features[2])
```

# Miscellaneous
Package verison: pytorch==1.7.0 torchvison==0.8.1
(https://github.com/pytorch/vision)

Download .pt files from (https://github.com/huyvnphan/PyTorch_CIFAR10)

Quantization of image using python library. Convert numpy array to image and quantize. I did not use this this time. 
```python
from PIL import Image  
import PIL 
img_as_img = Image.fromarray(a)
img_as_img.show()
#Quantize image which has 16 colors
im1 = img_as_img.quantize(16)  
```

Using hook in pytorch can inspect input and output from cetrain layer
```python
def printact(self, input, output):
     # input is a tuple not a tensor so it needs to be indexed.
     # output is a Tensor. output.data is the Tensor we are interested
     print('Inside ' + self.__class__.__name__ + ' forward')
     # input[0] is a tensor
     print('output max:', torch.max(input[0]).item())
     print('output min:', torch.min(input[0]).item())

my_model = vgg16_bn(pretrained=True)
my_model.eval() # for evaluation
data = CIFAR10Data(args)
dataloader = data.test_dataloader()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

my_model.features[5].register_forward_hook(printact)
....model inference....
```
If only one processor in CPU is used, don't forget to change num_workers to 0. Batch_size = 1 so inference happen with one image at a time. 
```python
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
```

Some useful links:

CIFAR custom dataset:

https://github.com/utkuozbulak/pytorch-custom-dataset-examples 
https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f

ReLU custom function:

https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
https://morioh.com/p/deaf2f23fbc6

pytorch hook:

https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
http://web.stanford.edu/~nanbhas/blog/forward-hook.html 

Iterate layers:

https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797
https://stackoverflow.com/questions/55875279/how-to-get-an-output-dimension-for-each-layer-of-the-neural-network-in-pytorch

Difference between nn.sequential() and writing out the layer directly:

https://stackoverflow.com/questions/55584747/torch-nn-sequential-vs-combination-of-multiple-torch-nn-linear
