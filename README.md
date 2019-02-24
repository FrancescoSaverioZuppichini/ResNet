
# ResNet
## Clean and scalable implementation of ResNet in Pytorch
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/ResNet/master/imagess/resnet34.png?token=APK83CAQ4DslctlEF0lFskydLIIX9MTbks5cepXFwA%3D%3D)

This is a ResNet implementation focus on scalable and customization. In addition to the classic ResNet model, we alsos provide Squeeze and Excitation blocks.

### Getting started

All classsic resnet models are avaiable by calling the factory methods from the package


```python
%load_ext autoreload
%autoreload 2
```


```python
from resnet import resnet18, resnet34, resnet50, resnet101, resnet50

model = resnet18(pretrained=False)
print(model)
```

    ResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
              (1): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
        )
      )
      (decoder): ResnetDecoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Linear(in_features=512, out_features=1000, bias=True)
      )
    )


By passing `pretrained=True`, the trained weights from `torchvision.models` will be loaded.

## Custom ResNet
To create a custom resnet you need to import the `ResNet` class


```python
from resnet import ResNet
```

#### Custom number of layers


```python

model = ResNet(depths=[1,1,1,1]) # resnet with 4 layers of 1 block each

print(model)
```

    ResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
        )
      )
      (decoder): ResnetDecoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Linear(in_features=512, out_features=1000, bias=True)
      )
    )


#### Custom number of filterss in each layer


```python
# resnet with 4 layers of 1 block each and custom filters
model = ResNet(depths=[1,1,1,1], blocks_sizes=[(8,8),(8,16),(16,32), (32,64)]) 

print(model)
```

    ResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 8, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
        )
      )
      (decoder): ResnetDecoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Linear(in_features=64, out_features=1000, bias=True)
      )
    )


#### Change the block

We implemented 4 blocks,

- `BasicBlock`
- `Bottleneck`
- `BasicBlockSE`
- `BottleneckSE`

You should always subclass `BasicBlock` in order to create a custom block and implement `blocks`. For example, `Bottleneck` is defined as follow


To use the basic resnet block, the one used in resnet18 and 34, you can import `BasicBlock` and pass it to `ResNet`.

```python
class Bottleneck(BasicBlock):
    expansion = 4

    def get_convs(self, in_planes, out_planes, stride, conv_block=conv_block3x3, *args, **kwargs):
        return nn.Sequential(
            conv_block3x3(in_planes, out_planes, kernel_size=1),
            conv_block3x3(out_planes, out_planes, kernel_size=3, stride=stride),
            conv_block3x3(out_planes, self.expanded, kernel_size=1),
        )
 ```
 
 You need to override the `get_convs`. The residual and the shortcat are automatically created for you.
 
`BasicBlockSE` and `BottleneckSE` uses the `SeModule` to weight each feature channel.

#### Custom input images


```python
model = resnet18(in_channel=1, pretrained=False) # for grey images
```

#### Subclassing Resnet


```python
import torch.nn as nn

class MyResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#         custom gate -> less aggressive conv and pool
        self.encoder.gate = nn.Sequential(
            nn.Conv2d(self.in_channel, self.encoder.blocks_sizes[0][0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.encoder.blocks_sizes[0][0]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )



    
my_resnet = MyResNet(depths=[1,1,1,1])

print(my_resnet)
```

    MyResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
        )
      )
      (decoder): ResnetDecoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Linear(in_features=512, out_features=1000, bias=True)
      )
    )


Since, `ResNet` also take an `Encoder` and `Decoder` parameter is it better to create custom classes to allow better modularity. Following the example from before, we want to change the `gate` conv to a 5x5


```python
from resnet import ResNetEncoder

class MyEncoder(ResNetEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#         custom gate -> less aggressive conv and pool
        self.gate = nn.Sequential(
            nn.Conv2d(self.in_channel, self.blocks_sizes[0][0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0][0]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

my_resnet = ResNet(depths=[1,1,1,1], encoder=MyEncoder)
print(my_resnet)
```

    ResNet(
      (encoder): MyEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
        )
      )
      (decoder): ResnetDecoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Linear(in_features=512, out_features=1000, bias=True)
      )
    )


#### Add a layer in the end

Just override `ResNetDecoder` and pass the `decoder` parameter to `ResNet`


```python
from resnet import ResnetDecoder

class MyDencoder(ResnetDecoder):
    def __init__(self, in_features, n_classes, *args, **kwargs):
        super().__init__(in_features, n_classes, *args, **kwargs)
#         you need to override decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Dropout(),
            nn.Linear(256, n_classes)
        )


my_resnet = ResNet(depths=[1,1,1,1], decoder=MyDencoder)
print(my_resnet)
```

    ResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01, inplace)
          (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (layers): ModuleList(
          (0): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
              )
            )
          )
          (1): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (2): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
          (3): ResNetLayer(
            (layer): Sequential(
              (0): BasicBlock(
                (convs): Sequential(
                  (0): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                  (1): Sequential(
                    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): ReLU()
                  )
                )
                (shortcut): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
          )
        )
      )
      (decoder): MyDencoder(
        (avg): AdaptiveAvgPool2d(output_size=(1, 1))
        (decoder): Sequential(
          (0): Linear(in_features=512, out_features=256, bias=True)
          (1): Dropout(p=0.5)
          (2): Linear(in_features=256, out_features=1000, bias=True)
        )
      )
    )


## Architecture

Resnet is composed by 5 main building blocks. From bottom to top

- `conv_block3x3` is the basic convolutional layer (conv -> batchnorm -> activation)
- `BasicBlock` is the basic residual block. `BottleNeckBlock` inherits from it
- `ResnetLayer` defines a residual layer by stacking multiples residual blocks together and by defining the shortcut
- `ResnetEncoder`, it is the head of the model, it stacks multiple `ResnetLayer` with a given depth
- `ResnetDecoder`, it is the tail of the model, it perform the average pooling and the classsification mapping

Following this phylosofy of composition, the `Resnet` class contains only the `ResnetEncoder` and the `ResnetDecoder`.

## References

1. Deep Residual Learning for Image Recognition He et al. [Paper](https://arxiv.org/pdf/1512.03385.pdf)
2. Identity Mappings in Deep Residual Networks He et al. [Paper](https://arxiv.org/pdf/1603.05027.pdf)
3. Squeeze-and-Excitation Networks [Paper](https://arxiv.org/abs/1709.01507)
