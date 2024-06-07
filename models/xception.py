import torch
import torch.nn as nn


__all__ = ['Xception', 'xception']


class BasicConv2d(nn.Module):
    def __init__(self, in_dims, out_dims, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, in_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class EntryFlow(nn.Module):
    def __init__(self, in_dims):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=32, kernel_size=3, stride=2, padding=1),
            BasicConv2d(in_dims=32, out_dims=64, kernel_size=3, stride=1, padding=0)
        )
        self.conv2_residual = nn.Sequential(
            DepthwiseSeparableConv(in_dims=64, out_dims=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=128, out_dims=128),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128)
        )
        self.conv3_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=128, out_dims=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=256, out_dims=256),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=256, out_dims=728),
            nn.BatchNorm2d(num_features=728),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=728),
            nn.BatchNorm2d(num_features=728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=728, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=728)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_shortcut(out_conv1) + self.conv2_residual(out_conv1)
        out_conv3 = self.conv3_shortcut(out_conv2) + self.conv3_residual(out_conv2)
        out_conv4 = self.conv4_shortcut(out_conv3) + self.conv4_residual(out_conv3)

        return out_conv4


class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.conv_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=728),
            nn.BatchNorm2d(num_features=728),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=728),
            nn.BatchNorm2d(num_features=728),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=728),
            nn.BatchNorm2d(num_features=728)
        )
        self.conv_shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv_shortcut(x) + self.conv_residual(x)

        return out


class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.conv1_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=728),
            nn.BatchNorm2d(num_features=728),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=728, out_dims=1024),
            nn.BatchNorm2d(num_features=1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv1_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(in_dims=1024, out_dims=1536),
            nn.BatchNorm2d(num_features=1536),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_dims=1536, out_dims=2048),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_conv1 = self.conv1_shortcut(x) + self.conv1_residual(x)
        out_conv2 = self.conv2(out_conv1)

        return out_conv2


class Xception(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Xception, self).__init__()
        self.entry = EntryFlow(in_dims=in_channels)
        self.middle = self._make_layers(block=MiddleFlow(), num_blocks=8)
        self.exit = ExitFlow()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        out = self.entry(x)
        out = self.middle(out)
        out = self.exit(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self,  block, num_blocks):
        layers = []
        for l in range(num_blocks):
            layers.append(block)

        return nn.Sequential(*layers)


def xception(**kwargs):
    return Xception(**kwargs)


if __name__ == '__main__':
    img_size = 224

    model = xception(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)