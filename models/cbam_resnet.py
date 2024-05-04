import torch
import torch.nn as nn


__all__ = ['CBAMResNet', 'cbam_resnet18', 'cbam_resnet34', 'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']


# set CBAM-ResNet configuration
cfgs = [64, 128, 256, 512]


class ChannelAttention(nn.Module):
    def __init__(self, in_dims, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, int(in_dims//reduction_ratio), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_dims//reduction_ratio), in_dims, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_avg = self.avgpool(x)
        out_avg = self.conv(out_avg)

        out_max = self.maxpool(x)
        out_max = self.conv(out_max)

        out = self.sigmoid(out_avg + out_max)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([x_avg, x_max], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_dims, out_dims, stride=1):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dims, out_dims*BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims*BasicBlock.expansion)
        )

        if stride != 1 or in_dims != out_dims*BasicBlock.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

        self.channel_attention = ChannelAttention(out_dims*BasicBlock.expansion)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out_shortcut = self.shortcut(x)

        out_residual = self.residual(x)
        out_residual = self.channel_attention(out_residual) * out_residual
        out_residual = self.spatial_attention(out_residual) * out_residual

        out = out_residual + out_shortcut
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dims, out_dims, stride=1):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dims, out_dims*Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims*Bottleneck.expansion)
        )

        if stride != 1 or in_dims != out_dims*Bottleneck.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

        self.channel_attention = ChannelAttention(out_dims*Bottleneck.expansion)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out_shortcut = self.shortcut(x)

        out_residual = self.residual(x)
        out_residual = self.channel_attention(out_residual) * out_residual
        out_residual = self.spatial_attention(out_residual) * out_residual

        out = out_residual + out_shortcut
        out = self.relu(out)

        return out


class CBAMResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes):
        super(CBAMResNet, self).__init__()
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


def cbam_resnet18(**kwargs):
    return CBAMResNet(block=BasicBlock, num_blocks=[2,2,2,2], **kwargs)


def cbam_resnet34(**kwargs):
    return CBAMResNet(block=BasicBlock, num_blocks=[3,4,6,3], **kwargs)


def cbam_resnet50(**kwargs):
    return CBAMResNet(block=Bottleneck, num_blocks=[3,4,6,3], **kwargs)


def cbam_resnet101(**kwargs):
    return CBAMResNet(block=Bottleneck, num_blocks=[3,4,23,3], **kwargs)


def cbam_resnet152(**kwargs):
    return CBAMResNet(block=Bottleneck, num_blocks=[3,8,36,3], **kwargs)


if __name__ == '__main__':
    img_size = 256

    model = cbam_resnet101(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)