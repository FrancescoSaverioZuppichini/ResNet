import torch
from resnet import *

dummy_x = torch.zeros((1, 3, 224, 224))

model = ResNet(3, depths=[1,1], blocks_sizes=[(64,64),(64,128)], preactivate=True, activation='leaky_relu')

model(dummy_x)

print(model)