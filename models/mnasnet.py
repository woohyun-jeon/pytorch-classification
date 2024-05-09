import torch
import torch.nn as nn


__all__ = ['MnasNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']


class MBConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, expand_ratio=6):
        super(MBConvBlock, self).__init__()
        if stride == 1 and in_dims == out_dims:
            self.use_residual = True
        else:
            self.use_residual = False

        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, int(in_dims*expand_ratio), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=int(in_dims*expand_ratio)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_dims*expand_ratio), int(in_dims*expand_ratio), kernel_size=kernel_size, stride=stride,
                      padding=int(kernel_size//2), groups=int(in_dims*expand_ratio), bias=False),
            nn.BatchNorm2d(num_features=int(in_dims*expand_ratio)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_dims*expand_ratio), out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims)
        )

    def forward(self, x):
        if self.use_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        return out


class MnasNet(nn.Module):
    def __init__(self, in_channels, num_classes, coef_width=1.0, p_dropout=0.5):
        super(MnasNet, self).__init__()
        dims = [32, 16, 24, 40, 80, 96, 192, 320, 1280]
        repeats = [3, 3, 3, 2, 4, 1]
        strides = [2, 2, 2, 1, 2, 1]
        kernels = [3, 5, 5, 3, 5, 3]

        dims = [int(dim*coef_width) for dim in dims]

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=dims[0]),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1, groups=dims[0], bias=False),
            nn.BatchNorm2d(num_features=dims[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(dims[0], dims[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=dims[1]),
        )
        self.inner_dims = dims[1]
        self.stage3 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[0], out_dims=dims[2],
                                        kernel_size=kernels[0], stride=strides[0], expand_ratio=3)
        self.stage4 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[1], out_dims=dims[3],
                                        kernel_size=kernels[1], stride=strides[1], expand_ratio=3)
        self.stage5 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[2], out_dims=dims[4],
                                        kernel_size=kernels[2], stride=strides[2], expand_ratio=6)
        self.stage6 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[3], out_dims=dims[5],
                                        kernel_size=kernels[3], stride=strides[3], expand_ratio=6)
        self.stage7 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[4], out_dims=dims[6],
                                        kernel_size=kernels[4], stride=strides[4], expand_ratio=6)
        self.stage8 = self._make_blocks(block=MBConvBlock, num_blocks=repeats[5], out_dims=dims[7],
                                        kernel_size=kernels[5], stride=strides[5], expand_ratio=6)
        self.stage9 = nn.Sequential(
            nn.Conv2d(dims[7], dims[8], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=dims[8]),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.classifier = nn.Linear(dims[8], num_classes)


    def forward(self, x):
        out = self.stage1(x)
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

    def _make_blocks(self, block, num_blocks, out_dims, kernel_size, stride, expand_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims=self.inner_dims, out_dims=out_dims, kernel_size=kernel_size, stride=stride,
                                expand_ratio=expand_ratio))
            self.inner_dims = out_dims

        return nn.Sequential(*layers)


def mnasnet0_5(**kwargs):
    return MnasNet(coef_width=0.5, **kwargs)


def mnasnet0_75(**kwargs):
    return MnasNet(coef_width=0.75, **kwargs)


def mnasnet1_0(**kwargs):
    return MnasNet(coef_width=1.0, **kwargs)


def mnasnet1_3(**kwargs):
    return MnasNet(coef_width=1.3, **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = mnasnet0_5(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)

    output = model(input)
    print(output.shape)