import torch
import torch.nn as nn


__all__ = ['WideResNet', 'wrn_40_10']


# set Wide ResNet configuration
cfgs = [16, 32, 64]


class WideBasicBlock(nn.Module):
    def __init__(self, in_dims, out_dims, stride, p_dropout=0.5):
        super(WideBasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if stride != 1 or in_dims != out_dims:
            self.shortcut = nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, in_channels, num_classes, p_dropout=0.5):
        super(WideResNet, self).__init__()
        N = int((depth - 4) / 6)
        self.p_dropout = p_dropout
        self.in_channels = cfgs[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cfgs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=cfgs[0]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = self._make_layers(out_dims=cfgs[0]*widen_factor, num_blocks=N, stride=1)
        self.conv3 = self._make_layers(out_dims=cfgs[1]*widen_factor, num_blocks=N, stride=2)
        self.conv4 = self._make_layers(out_dims=cfgs[2]*widen_factor, num_blocks=N, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(cfgs[2]*widen_factor, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self, out_dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(WideBasicBlock(self.in_channels, out_dims, stride, self.p_dropout))
            self.in_channels = out_dims

        return nn.Sequential(*layers)


def wrn_40_10(**kwargs):
    return WideResNet(depth=40, widen_factor=10, **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = wrn_40_10(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)