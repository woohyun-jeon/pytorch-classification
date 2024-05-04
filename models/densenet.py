import torch
import torch.nn as nn


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264']


class Bottleneck(nn.Module):
    def __init__(self, in_dims, growth_rate):
        super(Bottleneck, self).__init__()
        inner_dims = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, inner_dims, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dims, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = torch.cat([self.shortcut(x), self.residual(x)], dim=1)

        return out


class Transition(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Transition, self).__init__()

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.downsample(x)

        return out

class DenseNet(nn.Module):
    def __init__(self, num_block_lists, growth_rate=12, reduction=0.5, in_channels=3, num_classes=1000):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=inner_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential()

        for i in range(len(num_block_lists)-1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(num_block_lists[i], inner_channels))
            inner_channels += growth_rate * num_block_lists[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(num_block_lists)-1), self._make_dense_block(num_block_lists[len(num_block_lists)-1], inner_channels))
        inner_channels += growth_rate * num_block_lists[len(num_block_lists)-1]
        self.features.add_module('bn', nn.BatchNorm2d(num_features=inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_dense_block(self, num_blocks, inner_channels):
        dense_block = nn.Sequential()
        for i in range(num_blocks):
            dense_block.add_module('bottleneck_layer_{}'.format(i), Bottleneck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate

        return dense_block


def densenet121(**kwargs):
    return DenseNet(num_block_lists=[6,12,24,6], **kwargs)


def densenet169(**kwargs):
    return DenseNet(num_block_lists=[6,12,32,32], **kwargs)


def densenet201(**kwargs):
    return DenseNet(num_block_lists=[6,12,48,32], **kwargs)


def densenet264(**kwargs):
    return DenseNet(num_block_lists=[6,12,64,48], **kwargs)


if __name__ == '__main__':
    img_size = 224
    model = densenet169(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)