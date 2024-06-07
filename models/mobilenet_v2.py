import torch
import torch.nn as nn


__all__ = ['MobileNetV2', 'mobilenet_v2']


class BasicConv2d(nn.Module):
    def __init__(self, in_dims, out_dims, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class InvertedBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dims, out_dims, expand_ratio, stride=1):
        super(InvertedBottleneck, self).__init__()
        hidden_dims = int(in_dims*expand_ratio)
        self.residual = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=hidden_dims, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=hidden_dims, out_dims=hidden_dims, kernel_size=3, stride=stride, padding=1, groups=hidden_dims),
            nn.Conv2d(in_channels=hidden_dims, out_channels=out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims)
        )

        if stride != 1 or in_dims != out_dims:
            self.shortcut = nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.relu(out)

        return out


class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, p_dropout=0.5):
        super(MobileNetV2, self).__init__()
        self.conv_init = BasicConv2d(in_dims=in_channels, out_dims=32, kernel_size=3, stride=2, padding=1)
        self.in_dims = 32
        self.features = nn.Sequential(
            self._make_layers(num_blocks=1, out_dims=16, expand_ratio=1, stride=1),
            self._make_layers(num_blocks=2, out_dims=24, expand_ratio=6, stride=2),
            self._make_layers(num_blocks=3, out_dims=32, expand_ratio=6, stride=2),
            self._make_layers(num_blocks=4, out_dims=64, expand_ratio=6, stride=2),
            self._make_layers(num_blocks=3, out_dims=96, expand_ratio=6, stride=1),
            self._make_layers(num_blocks=3, out_dims=160, expand_ratio=6, stride=2),
            self._make_layers(num_blocks=1, out_dims=320, expand_ratio=6, stride=1),
        )
        self.conv_last = BasicConv2d(in_dims=320, out_dims=1280, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        out = self.conv_init(x)
        out = self.features(out)
        out = self.conv_last(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def _make_layers(self, num_blocks, out_dims, expand_ratio, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(InvertedBottleneck(in_dims=self.in_dims, out_dims=out_dims, expand_ratio=expand_ratio, stride=_stride))
            self.in_dims = out_dims

        return nn.Sequential(*layers)


def mobilenet_v2(**kwargs):
    return MobileNetV2(**kwargs)


if __name__ == '__main__':
    img_size = 224

    model = mobilenet_v2(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)