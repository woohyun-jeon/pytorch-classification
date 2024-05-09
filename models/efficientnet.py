import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['EfficientNet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


class SEBlock(nn.Module):
    def __init__(self, in_dims, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_dims, int(in_dims//reduction_ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_dims//reduction_ratio), in_dims, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_squeeze = self.squeeze(x).view(x.size(0), x.size(1))
        out_excitation = self.excitation(out_squeeze).view(x.size(0), x.size(1), 1, 1)
        out = x * out_excitation.expand_as(x)

        return out


class MBConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, expand_ratio=6, reduction_ratio=4):
        super(MBConvBlock, self).__init__()
        if expand_ratio > 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_dims, int(in_dims*expand_ratio), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=int(in_dims*expand_ratio)),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Sequential()

        self.conv = nn.Sequential(
            nn.Conv2d(int(in_dims*expand_ratio), int(in_dims*expand_ratio), kernel_size=kernel_size, stride=stride,
                      padding=int(kernel_size//2), groups=int(in_dims*expand_ratio), bias=False),
            nn.BatchNorm2d(num_features=int(in_dims*expand_ratio)),
            nn.SiLU(inplace=True),

            SEBlock(int(in_dims*expand_ratio), reduction_ratio=reduction_ratio),

            nn.Conv2d(int(in_dims*expand_ratio), out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims)
        )

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.conv(out)

        return out


class EfficientNet(nn.Module):
    def __init__(self, in_channels, num_classes, coef_width, coef_depth, up_scale, reduction_ratio, p_dropout=0.5):
        super(EfficientNet, self).__init__()
        dims = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernels = [3, 3, 5, 3, 5, 5, 3]

        dims = [int(dim*coef_width) for dim in dims]
        repeats = [int(repeat*coef_depth) for repeat in repeats]

        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=dims[0]),
            nn.ReLU(inplace=True)
        )
        self.inner_dims = dims[0]
        self.stage2 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[0], out_dims=dims[1],
                                        kernel_size=kernels[0], stride=strides[0], expand_ratio=1, reduction_ratio=reduction_ratio)
        self.stage3 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[1], out_dims=dims[2],
                                        kernel_size=kernels[1], stride=strides[1], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage4 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[2], out_dims=dims[3],
                                        kernel_size=kernels[2], stride=strides[2], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage5 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[3], out_dims=dims[4],
                                        kernel_size=kernels[3], stride=strides[3], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage6 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[4], out_dims=dims[5],
                                        kernel_size=kernels[4], stride=strides[4], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage7 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[5], out_dims=dims[6],
                                        kernel_size=kernels[5], stride=strides[5], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage8 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[6], out_dims=dims[7],
                                        kernel_size=kernels[6], stride=strides[6], expand_ratio=6, reduction_ratio=reduction_ratio)
        self.stage9 = nn.Sequential(
            nn.Conv2d(dims[7], dims[8], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=dims[8]),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.classifier = nn.Linear(dims[8], num_classes)

    def forward(self, x):
        out = self.upsample(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)
        out = self.stage9(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def _make_blocks(self, block, num_blocks, out_dims, kernel_size, stride, expand_ratio, reduction_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims=self.inner_dims, out_dims=out_dims, kernel_size=kernel_size, stride=stride,
                                expand_ratio=expand_ratio, reduction_ratio=reduction_ratio))
            self.inner_dims = out_dims

        return nn.Sequential(*layers)


def efficientnet_b0(img_size, **kwargs):
    return EfficientNet(coef_width=1.0, coef_depth=1.0, up_scale=224/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b1(img_size, **kwargs):
    return EfficientNet(coef_width=1.0, coef_depth=1.1, up_scale=240/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b2(img_size, **kwargs):
    return EfficientNet(coef_width=1.1, coef_depth=1.2, up_scale=260/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b3(img_size, **kwargs):
    return EfficientNet(coef_width=1.2, coef_depth=1.4, up_scale=300/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b4(img_size, **kwargs):
    return EfficientNet(coef_width=1.4, coef_depth=1.8, up_scale=380/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b5(img_size, **kwargs):
    return EfficientNet(coef_width=1.6, coef_depth=2.2, up_scale=456/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b6(img_size, **kwargs):
    return EfficientNet(coef_width=1.8, coef_depth=2.6, up_scale=528/img_size, reduction_ratio=4, **kwargs)

def efficientnet_b7(img_size, **kwargs):
    return EfficientNet(coef_width=2.0, coef_depth=3.1, up_scale=600/img_size, reduction_ratio=4, **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = efficientnet_b0(in_channels=3, num_classes=1000, img_size=img_size)

    input = torch.randn(4, 3, img_size, img_size)

    output = model(input)
    print(output.shape)