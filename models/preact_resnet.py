import torch
import torch.nn as nn


__all__ = ['PreActResNet', 'preact_resnet18', 'preact_resnet34', 'preact_resnet50', 'preact_resnet101', 'preact_resnet152']


# set Pre Activation ResNet configuration
cfgs = [64, 128, 256, 512]


class PreActBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_dims, out_dims, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims*PreActBasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if stride != 1 or in_dims != out_dims*PreActBasicBlock.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*PreActBasicBlock.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)

        return out


class PreActBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dims, out_dims, stride=1):
        super(PreActBottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, bias=False),

            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims*PreActBottleneck.expansion, kernel_size=1, stride=1, bias=False),
        )

        if stride != 1 or in_dims != out_dims*PreActBottleneck.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*PreActBottleneck.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes):
        super(PreActResNet, self).__init__()
        self.in_channels = cfgs[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cfgs[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=cfgs[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layers(block, num_blocks[0], out_dims=cfgs[0], stride=1)
        self.conv3_x = self._make_layers(block, num_blocks[1], out_dims=cfgs[1], stride=2)
        self.conv4_x = self._make_layers(block, num_blocks[2], out_dims=cfgs[2], stride=2)
        self.conv5_x = self._make_layers(block, num_blocks[3], out_dims=cfgs[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(cfgs[3]*block.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self, block, num_blocks, out_dims, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_dims, stride))
            self.in_channels = out_dims*block.expansion

        return nn.Sequential(*layers)


def preact_resnet18(**kwargs):
    return PreActResNet(block=PreActBasicBlock, num_blocks=[2,2,2,2], **kwargs)


def preact_resnet34(**kwargs):
    return PreActResNet(block=PreActBasicBlock, num_blocks=[3,4,6,3], **kwargs)


def preact_resnet50(**kwargs):
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,4,6,3], **kwargs)


def preact_resnet101(**kwargs):
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,4,23,3], **kwargs)


def preact_resnet152(**kwargs):
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,8,36,3], **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = preact_resnet152(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)