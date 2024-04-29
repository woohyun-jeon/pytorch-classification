from collections import namedtuple
from typing import Optional, Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "InceptionV3Outputs",
    "InceptionV3",
    "BasicConv2d", "InceptionA", "InceptionB", "InceptionC", "InceptionD", "InceptionE", "InceptionAux",
    "inception_v3",
]

# According to the writing of the official library of Torchvision
InceptionV3Outputs = namedtuple("InceptionV3Outputs", ["logits", "aux_logits", ])
InceptionV3Outputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}


class InceptionV3(nn.Module):
    __constants__ = ["aux_logits"]

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1000,
            aux_logits: bool = False,
            dropout: float = 0.5,
    ) -> None:
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits

        self.Conv2d_1a_3x3 = BasicConv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.Conv2d_2a_3x3 = BasicConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(in_channels=192, pool_features=32)
        self.Mixed_5c = InceptionA(in_channels=256, pool_features=64)
        self.Mixed_5d = InceptionA(in_channels=288, pool_features=64)

        self.Mixed_6a = InceptionB(in_channels=288)

        self.Mixed_6b = InceptionC(in_channels=768, channels_7x7=128)
        self.Mixed_6c = InceptionC(in_channels=768, channels_7x7=160)
        self.Mixed_6d = InceptionC(in_channels=768, channels_7x7=160)
        self.Mixed_6e = InceptionC(in_channels=768, channels_7x7=192)

        if aux_logits:
            self.aux = InceptionAux(in_channels=768, num_classes=num_classes)
        else:
            self.aux = None

        self.Mixed_7a = InceptionD(in_channels=768)
        self.Mixed_7b = InceptionE(in_channels=1280)
        self.Mixed_7c = InceptionE(in_channels=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor):
        out = self.Conv2d_1a_3x3(x)
        out = self.Conv2d_2a_3x3(out)
        out = self.Conv2d_2b_3x3(out)
        out = self.maxpool1(out)
        out = self.Conv2d_3b_1x1(out)
        out = self.Conv2d_4a_3x3(out)
        out = self.maxpool2(out)

        out = self.Mixed_5b(out)
        out = self.Mixed_5c(out)
        out = self.Mixed_5d(out)

        out = self.Mixed_6a(out)

        out = self.Mixed_6b(out)
        out = self.Mixed_6c(out)
        out = self.Mixed_6d(out)
        out = self.Mixed_6e(out)

        aux1: Optional[Tensor] = None
        if self.aux is not None:
            if self.training:
                aux1 = self.aux(out)

        out = self.Mixed_7a(out)

        out = self.Mixed_7b(out)
        out = self.Mixed_7c(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        aux2 = self.fc(out)

        if self.training and self.aux_logits:
            return InceptionV3Outputs(aux2, aux1)
        else:
            return aux2

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class InceptionA(nn.Module):
    # Inception module in Figure 5
    def __init__(
            self,
            in_channels: int,
            pool_features: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.branch5x5_1 = BasicConv2d(in_channels=in_channels, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.branch5x5_2 = BasicConv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_channels=in_channels, out_channels=pool_features, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        out = torch.cat(out, dim=1)

        return out


class InceptionB(nn.Module):
    # Grid size reduction block between the Inception modules in Figures 5 and 6
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels=in_channels, out_channels=384, kernel_size=3, stride=2, padding=0)

        self.branch3x3dbl_1 = BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.maxpool(x)

        out = [branch3x3, branch3x3dbl, branch_pool]
        out = torch.cat(out, dim=1)

        return out


class InceptionC(nn.Module):
    # Inception module in Figure 6
    def __init__(
            self,
            in_channels: int,
            channels_7x7: int,
    ) -> None:
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels=in_channels, out_channels=192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels=in_channels, out_channels=channels_7x7, kernel_size=1, stride=1, padding=0)
        self.branch7x7_2 = BasicConv2d(in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(in_channels=channels_7x7, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels=in_channels, out_channels=channels_7x7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(in_channels=channels_7x7, out_channels=192, kernel_size=(1, 7), stride=1, padding=(0, 3))

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        out = torch.cat(out, dim=1)

        return out


class InceptionD(nn.Module):
    # Grid size reduction block between the Inception modules in Figures 6 and 7
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2 = BasicConv2d(in_channels=192, out_channels=320, kernel_size=3, stride=2, padding=0)

        self.branch7x7x3_1 = BasicConv2d(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.branch7x7x3_2 = BasicConv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool(x)

        out = [branch3x3, branch7x7x3, branch_pool]
        out = torch.cat(out, dim=1)

        return out


class InceptionE(nn.Module):
    # Inception module in Figure 7
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels=in_channels, out_channels=320, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = BasicConv2d(in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2a = BasicConv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels=in_channels, out_channels=448, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_channels=448, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_channels=in_channels, out_channels=192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(tensors=[self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], dim=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat(tensors=[self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], dim=1)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        out = torch.cat(out, dim=1)

        return out


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
    ) -> None:
        super().__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv1 = BasicConv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=768, out_features=num_classes)

        self.conv1.stddev = 0.01
        self.fc.stddev = 0.001

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool1(x)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.avgpool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def inception_v3(**kwargs: Any) -> InceptionV3:
    model = InceptionV3(**kwargs)

    return model
