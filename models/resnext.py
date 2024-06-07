import torch
import torch.nn as nn


__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


# set ResNeXt configuration
base_width = 64
depth = 4
cardinality = 32 # the number of groups


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dims, out_dims, stride=1):
        super(Bottleneck, self).__init__()
        C = cardinality
        D = int(depth * out_dims / base_width)
        self.residual = nn.Sequential(
            nn.Conv2d(in_dims, int(C*D), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=int(C*D)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(C*D), int(C*D), kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
            nn.BatchNorm2d(num_features=int(C*D)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(C*D), out_dims*Bottleneck.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims*Bottleneck.expansion)
        )

        if stride != 1 or in_dims != out_dims*Bottleneck.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes):
        super(ResNeXt, self).__init__()
        self.in_channels = base_width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=base_width),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layers(block, num_blocks[0], out_dims=int(base_width*1), stride=1)
        self.conv3_x = self._make_layers(block, num_blocks[1], out_dims=int(base_width*2), stride=2)
        self.conv4_x = self._make_layers(block, num_blocks[2], out_dims=int(base_width*3), stride=2)
        self.conv5_x = self._make_layers(block, num_blocks[3], out_dims=int(base_width*4), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(int(base_width*4)*block.expansion, num_classes)

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


def resnext50(**kwargs):
    return ResNeXt(block=Bottleneck, num_blocks=[3,4,6,3], **kwargs)


def resnext101(**kwargs):
    return ResNeXt(block=Bottleneck, num_blocks=[3,4,23,3], **kwargs)


def resnext152(**kwargs):
    return ResNeXt(block=Bottleneck, num_blocks=[3,8,36,3], **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = resnext152(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)