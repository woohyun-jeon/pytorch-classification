import torch
import torch.nn as nn


__all__ = ['MobileNetV1', 'mobilenet_v1']


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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dims, out_dims, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = BasicConv2d(in_dims=in_dims, out_dims=in_dims, kernel_size=3, stride=stride, padding=1, groups=in_dims)
        self.pointwise = BasicConv2d(in_dims=in_dims, out_dims=out_dims, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes, width_multiplier):
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(in_dims=in_channels, out_dims=int(32*width_multiplier), kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(in_dims=int(32*width_multiplier), out_dims=int(64*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(64*width_multiplier), out_dims=int(128*width_multiplier), stride=2),
            DepthwiseSeparableConv(in_dims=int(128*width_multiplier), out_dims=int(128*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(128*width_multiplier), out_dims=int(256*width_multiplier), stride=2),
            DepthwiseSeparableConv(in_dims=int(256*width_multiplier), out_dims=int(256*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(256*width_multiplier), out_dims=int(512*width_multiplier), stride=2),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(512*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(512*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(512*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(512*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(512*width_multiplier), stride=1),
            DepthwiseSeparableConv(in_dims=int(512*width_multiplier), out_dims=int(1024*width_multiplier), stride=2),
            DepthwiseSeparableConv(in_dims=int(1024*width_multiplier), out_dims=int(1024*width_multiplier), stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(in_features=int(1024*width_multiplier), out_features=num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


def mobilenet_v1(**kwargs):
    return MobileNetV1(**kwargs)


if __name__ == '__main__':
    img_size = 224

    model = mobilenet_v1(in_channels=3, num_classes=1000, width_multiplier=1.0)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)