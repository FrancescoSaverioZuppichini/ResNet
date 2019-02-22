import torch
import torch.nn as nn

from torchvision import models

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, *args, **kwargs):
        super().__init__()
        padding = ((in_channels - 1) * (stride - 1) + 1 * (kernel_size - 1)) // 2

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding)
        )

    def forward(self, x):
        return self.net(x)


class SEModule(nn.Module):
    def __init__(self, n_features, ratio=16, *args, **kwargs):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Linear(n_features, n_features // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(n_features // ratio, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)  # flat
        out = self.se(out).view(b, c, 1, 1)

        return x * out


def conv_block(in_planes, out_planes, conv_layer=nn.Conv2d, kernel_size=3, padding=None, preactivated=False, stride=1,
               **kwargs):
    padding = kernel_size // 2 if not padding else padding

    if preactivated:
        conv_block = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(negative_slope=0.1),
            conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                       **kwargs)
        )
    else:
        conv_block = nn.Sequential(
            conv_layer(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                       **kwargs),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(),
        )

    return conv_block


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        self.in_planes, self.out_planes, self.conv_layer, self.stride = in_planes, out_planes, conv_layer, stride

        self.block = self.blocks(in_planes, out_planes, conv_layer, stride=stride, *args, **kwargs)

        self.shortcut = self.get_shortcut() if self.in_planes != self.expanded else None

    @property
    def expanded(self):
        return self.out_planes * self.expansion

    def get_shortcut(self):
        return nn.Sequential(
            self.conv_layer(self.in_planes, self.out_planes, kernel_size=1,
                       stride=self.stride, bias=False),
            nn.BatchNorm2d(self.out_planes),
        )

    def blocks(self, in_planes, out_planes, conv_layer, stride, *args, **kwargs):
        return nn.Sequential(
            conv_block(self.in_planes, out_planes, conv_layer, stride=stride, *args, **kwargs),
            conv_block(out_planes, out_planes, conv_layer, *args, **kwargs),
        )


    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)

        out = out + residual

        return out

class Bottleneck(BasicBlock):
    expansion = 4

    def blocks(self, in_planes, out_planes, conv_layer, stride, *args, **kwargs):
        return nn.Sequential(
            conv_block(in_planes, out_planes, conv_layer, kernel_size=1),
            conv_block(out_planes, out_planes, conv_layer, kernel_size=3, stride=stride),
            conv_block(out_planes, self.expanded, conv_layer, kernel_size=1),
        )


class BasicBlockSE(BasicBlock):
    def __init__(self, in_planes, out_planes, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__(in_planes, out_planes, conv_layer=conv_layer, *args, **kwargs)
        self.se = SEModule(out_planes)

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)
        out = self.se(out)
        out += residual

        return out


class BottleneckSE(Bottleneck):
    expansion = 4

    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super().__init__(in_planes, out_planes, *args, **kwargs)
        self.se = SEModule(out_planes * self.expansion)

    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = self.block(x)
        out = self.se(out)

        out.add_(residual)

        return out


class ResNetLayer(nn.Module):
    """
    This class represent a layer of ResNet, it stacks a number of block
    equal to the depth parameter. If the `in_planes` and `out_planes` are different,
    for example 64 features in and 128 features out, the first layer will use a stride of two,
    quoting from the paper: 'We perform downsampling directly by convolutional layers that have a stride of 2.'
    """
    def __init__(self, in_planes, out_planes, depth, block=BasicBlock, *args, **kwargs):
        super().__init__()
        # if inputs == outputs then stride = 1, e.g 64==64 (first block)
        stride = 1 if in_planes == out_planes else 2

        expansion = block.expansion

        in_planes = in_planes * expansion
        # create the layer by directly instantiate the first block with the correct stride and then
        # if needed create all the others blocks
        self.layer = nn.Sequential(
            block(in_planes, out_planes, stride=stride, *args, **kwargs),
            *[block(out_planes * block.expansion, out_planes, *args, **kwargs) for _ in range(max(0, depth - 1))],
        )

    def forward(self, x):
        out = self.layer(x)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, in_channel, depths, blocks=BasicBlock, blocks_sizes=None, conv_layer=nn.Conv2d, *args, **kwargs):
        super().__init__()

        self.gate = nn.Sequential(
            conv_layer(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.blocks_sizes = [(64, 64), (64, 128), (128, 256), (256, 512)]

        if type(blocks) is not list: blocks = [blocks] * len(self.blocks_sizes)

        self.blocks = blocks

        if blocks_sizes is None: blocks_sizes = self.blocks_sizes

        self.layers = nn.ModuleList([
            ResNetLayer(in_c, out_c, depth=depths[i], block=blocks[i], conv_layer=conv_layer, *args, **kwargs)
            for i, (in_c, out_c) in enumerate(blocks_sizes)
        ])

        self.initialise(self.modules())

    @staticmethod
    def initialise(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.gate(x)

        for layer in self.layers:
            x = layer(x)

        return x

class ResnetDecoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    """
    ResNet https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, in_channel, depths, blocks=BasicBlock, conv_layer=nn.Conv2d, n_classes=1000, encoder=ResNetEncoder, decoder=ResnetDecoder, *args, **kwargs):
        super().__init__()
        self.encoder = encoder(in_channel,depths, blocks, conv_layer=conv_layer, *args, **kwargs)
        self.decoder = decoder(self.encoder.blocks_sizes[-1][1] * self.encoder.blocks[-1].expansion, n_classes)


    def forward(self, x):
        return self.decoder(self.encoder(x))


# resnet-tiny = [1, 2, 3, 2]

def resnet18(in_channel, block=BasicBlock, resnet=ResNet, pretrained=False, *args, **kwargs):
    model = resnet(in_channel, [2, 2, 2, 2], block, *args, **kwargs)

    if pretrained:
        print('loading pretrained weights...')
        restore(models.resnet18(True), model)

    return model

def resnet34(in_channel, block=BasicBlock, pretrained=False, resnet=ResNet, **kwargs):
    model = resnet(in_channel, [3, 4, 6, 3], block, **kwargs)

    if pretrained:
        print('loading pretrained weights...')
        restore(models.resnet34(True), model)

    return model

def resnet50(in_channel, block=Bottleneck, **kwargs):
    model = ResNet(in_channel, [3, 4, 6, 3], block, **kwargs)

    return model

def resnet101(in_channel, block=Bottleneck, **kwargs):
    model = ResNet(in_channel, [3, 4, 23, 3], block, **kwargs)

    return model

def resnet152(in_channel, block=Bottleneck, pretrained=False, **kwargs):

    model = ResNet(in_channel, [3, 8, 36, 3], block, **kwargs)

    return model

def restore(source, target):
    pre_trained_layers = [source.layer1, source.layer2, source.layer3, source.layer4]

    for i,pre_trained_layer in enumerate(pre_trained_layers):
        layer = target.encoder.layers[i]
        p_t_convs = [m for m in  pre_trained_layer.modules() if isinstance(m, nn.Conv2d)]
        p_t_bns = [m for m in  pre_trained_layer.modules() if isinstance(m, nn.BatchNorm2d)]

        convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
        bns = [m for m in layer.modules() if isinstance(m, nn.BatchNorm2d)]

        for p_t_conv, conv in zip(p_t_convs, convs):
            conv.load_state_dict(p_t_conv.state_dict())

        for p_t_bn, bn in zip(p_t_bns, bns):
            bn.load_state_dict(p_t_bn.state_dict())


# resnet = ResNet(1, [2,2,2,2])
#
# print(resnet)