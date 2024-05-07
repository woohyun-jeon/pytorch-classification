import torch
import torch.nn as nn


__all__ = ['GoogLeNet', 'googlenet']


class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(in_dims=in_channels, out_dims=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(in_dims=64, out_dims=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = BasicConv2d(in_dims=64, out_dims=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(in_dims=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_dims=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(in_dims=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_dims=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_dims=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_dims=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_dims=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = Inception(in_dims=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_dims=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

        # Initialize neural network weights
        self.initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, val=1)
                nn.init.constant_(module.bias, val=0)


class BasicConv2d(nn.Module):
    def __init__(self, in_dims, out_dims, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=out_dims, eps=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class Inception(nn.Module):
    def __init__(self, in_dims, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_dims=in_dims, out_dims=ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=ch3x3red, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=ch3x3red, out_dims=ch3x3, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=ch5x5red, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=ch5x5red, out_dims=ch5x5, kernel_size=5, stride=1, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_dims=in_dims, out_dims=pool_proj, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out_branch1 = self.branch1(x)
        out_branch2 = self.branch2(x)
        out_branch3 = self.branch3(x)
        out_branch4 = self.branch4(x)

        out = torch.cat([out_branch1, out_branch2, out_branch3, out_branch4], dim=1)

        return out


class InceptionAux(nn.Module):
    def __init__(self, in_dims, num_classes, dropout=0.7):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4,4))
        self.conv = BasicConv2d(in_dims=in_dims, out_dims=128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.classifier1 = nn.Linear(in_features=2048, out_features=1024)
        self.classifier2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.classifier2(out)

        return out


def googlenet(**kwargs):
    model = GoogLeNet(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 224
    model = googlenet(in_channels=3, num_classes=1000, dropout=0.7)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)