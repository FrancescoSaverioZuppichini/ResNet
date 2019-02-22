import torch
from resnet import *

dummy_x = torch.zeros((1, 3, 224, 224))

model = resnet18(3)

model(dummy_x)

print(model)