from collections import namedtuple
from typing import Optional, Tuple, Any

import torch
from torch import Tensor
from torch import nn

from config import config_GoogLeNet as config

__all__ = [
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

# According to the writing of the official library of Torchvision
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}


class GoogLeNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1000,
            aux_logits: bool = False,
            dropout: float = 0.4,
            dropout_aux: float = 0.7,
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)

        if aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes, dropout=dropout_aux)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        # Initialize neural network weights
        self.initialize_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        aux3 = self.fc(out)

        if self.training and self.aux_logits:
            return GoogLeNetOutputs(aux3, aux2, aux1)
        else:
            return aux3

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, val=1)
                nn.init.constant_(module.bias, val=0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]

        out = torch.cat(out, 1)

        return out


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dropout: float = 0.7,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def googlenet(**kwargs: Any) -> GoogLeNet:
    model = GoogLeNet(**kwargs)

    return model
