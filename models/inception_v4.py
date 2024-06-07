import torch
import torch.nn as nn


__all__ = ['InceptionV4', 'inception_v4']


class InceptionV4(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5, k=192, l=224, m=256, n=384):
        super(InceptionV4, self).__init__()
        self.stem = InceptionStem(in_dims=in_channels)
        self.inception_a = self._make_layers(in_dims=384, out_dims=384, block=InceptionA, num_blocks=4)
        self.reduction_a = ReductionA(in_dims=384, k=k, l=l, m=m, n=n)
        self.inception_b = self._make_layers(in_dims=self.reduction_a.out_dims, out_dims=1024, block=InceptionB, num_blocks=7)
        self.reduction_b = ReductionB(in_dims=1024)
        self.inception_c = self._make_layers(in_dims=self.reduction_b.out_dims, out_dims=1536, block=InceptionC, num_blocks=3)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = nn.Linear(in_features=1536, out_features=num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_a(out)
        out = self.reduction_a(out)
        out = self.inception_b(out)
        out = self.reduction_b(out)
        out = self.inception_c(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layers(self, in_dims, out_dims, block, num_blocks):
        layers = []
        for l in range(num_blocks):
            layers.append(block(in_dims=in_dims))
            in_dims = out_dims

        return nn.Sequential(*layers)


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


class InceptionStem(nn.Module):
    def __init__(self, in_dims):
        super(InceptionStem, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=32, kernel_size=3, stride=2, padding=0),
            BasicConv2d(in_dims=32, out_dims=32, kernel_size=3, stride=1, padding=0),
            BasicConv2d(in_dims=32, out_dims=64, kernel_size=3, stride=1, padding=1),
        )

        self.branch3x3_conv = BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=2, padding=0)
        self.branch3x3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(in_dims=96+64, out_dims=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=0)
        )
        self.branch7x7b = nn.Sequential(
            BasicConv2d(in_dims=96+64, out_dims=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=64, out_dims=64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(in_dims=64, out_dims=64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=0)
        )

        self.branch_poola = BasicConv2d(in_dims=96+96, out_dims=192, kernel_size=3, stride=2, padding=0)
        self.branch_poolb = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_branch3x3_conv = self.branch3x3_conv(out_conv1)
        out_branch3x3_pool = self.branch3x3_pool(out_conv1)
        out_branch3x3 = torch.cat([out_branch3x3_conv, out_branch3x3_pool], dim=1)

        out_branch7x7a = self.branch7x7a(out_branch3x3)
        out_branch7x7b = self.branch7x7b(out_branch3x3)
        out_branch7x7 = torch.cat([out_branch7x7a, out_branch7x7b], dim=1)

        out_branch_poola = self.branch_poola(out_branch7x7)
        out_branch_poolb = self.branch_poolb(out_branch7x7)
        out_branch_pool = torch.cat([out_branch_poola, out_branch_poolb], dim=1)

        return out_branch_pool


class InceptionA(nn.Module):
    def __init__(self, in_dims):
        super(InceptionA, self).__init__()
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_dims=in_dims, out_dims=96, kernel_size=1, stride=1, padding=0)
        )
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=96, kernel_size=1, stride=1, padding=0)

        self.branch3x3a = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3x3b = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_dims=96, out_dims=96, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out_branch_pool = self.branch_pool(x)
        out_branch1x1 = self.branch1x1(x)
        out_branch3x3a = self.branch3x3a(x)
        out_branch3x3b = self.branch3x3b(x)

        out = torch.cat([out_branch_pool, out_branch1x1, out_branch3x3a, out_branch3x3b], dim=1)

        return out


class ReductionA(nn.Module):
    def __init__(self, in_dims, k, l, m, n):
        super(ReductionA, self).__init__()
        self.branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_b = BasicConv2d(in_dims=in_dims, out_dims=n, kernel_size=3, stride=2, padding=0)
        self.branch_c = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=k, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=k, out_dims=l, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_dims=l, out_dims=m, kernel_size=3, stride=2, padding=0)
        )
        self.out_dims = in_dims + m + n

    def forward(self, x):
        out_branch_a = self.branch_a(x)
        out_branch_b = self.branch_b(x)
        out_branch_c = self.branch_c(x)

        out = torch.cat([out_branch_a, out_branch_b, out_branch_c], dim=1)

        return out


class InceptionB(nn.Module):
    def __init__(self, in_dims):
        super(InceptionB, self).__init__()
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_dims=in_dims, out_dims=128, kernel_size=1, stride=1, padding=0)
        )
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=384, kernel_size=1, stride=1, padding=0)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=192, out_dims=224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(in_dims=224, out_dims=256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=192, out_dims=192, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(in_dims=192, out_dims=224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(in_dims=224, out_dims=224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(in_dims=224, out_dims=256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

    def forward(self, x):
        out_branch_pool = self.branch_pool(x)
        out_branch1x1 = self.branch1x1(x)
        out_branch7x7a = self.branch7x7a(x)
        out_branch7x7b = self.branch7x7b(x)

        out = torch.cat([out_branch_pool, out_branch1x1, out_branch7x7a, out_branch7x7b], dim=1)

        return out


class ReductionB(nn.Module):
    def __init__(self, in_dims):
        super(ReductionB, self).__init__()
        self.branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_b = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=192, out_dims=192, kernel_size=3, stride=2, padding=0),
        )
        self.branch_c = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=256, out_dims=256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(in_dims=256, out_dims=320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(in_dims=320, out_dims=320, kernel_size=3, stride=2, padding=0)
        )
        self.out_dims = in_dims + 320 + 192

    def forward(self, x):
        out_branch_a = self.branch_a(x)
        out_branch_b = self.branch_b(x)
        out_branch_c = self.branch_c(x)

        out = torch.cat([out_branch_a, out_branch_b, out_branch_c], dim=1)

        return out

class InceptionC(nn.Module):
    def __init__(self, in_dims):
        super(InceptionC, self).__init__()
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_dims=in_dims, out_dims=256, kernel_size=1, stride=1, padding=0)
        )
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=256, kernel_size=1, stride=1, padding=0)

        self.branch3x3a1 = BasicConv2d(in_dims=in_dims, out_dims=384, kernel_size=1, stride=1, padding=0)
        self.branch3x3a2_1 = BasicConv2d(in_dims=384, out_dims=256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch3x3a2_2 = BasicConv2d(in_dims=384, out_dims=256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3x3b1 = nn.Sequential(
            BasicConv2d(in_dims=in_dims, out_dims=384, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_dims=384, out_dims=448, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(in_dims=448, out_dims=512, kernel_size=(3,1), stride=1, padding=(1,0))
        )
        self.branch3x3b2_1 = BasicConv2d(in_dims=512, out_dims=256, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch3x3b2_2 = BasicConv2d(in_dims=512, out_dims=256, kernel_size=(1,3), stride=1, padding=(0,1))

    def forward(self, x):
        out_branch_pool = self.branch_pool(x)

        out_branch1x1 = self.branch1x1(x)

        out_branch3x3a1 = self.branch3x3a1(x)
        out_branch3x3a2_1 = self.branch3x3a2_1(out_branch3x3a1)
        out_branch3x3a2_2 = self.branch3x3a2_2(out_branch3x3a1)

        out_branch3x3b1 = self.branch3x3b1(x)
        out_branch3x3b2_1 = self.branch3x3b2_1(out_branch3x3b1)
        out_branch3x3b2_2 = self.branch3x3b2_2(out_branch3x3b1)

        out = torch.cat([out_branch_pool, out_branch1x1, out_branch3x3a2_1, out_branch3x3a2_2,
                                out_branch3x3b2_1, out_branch3x3b2_2], dim=1)

        return out


def inception_v4(**kwargs):
    model = InceptionV4(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 224
    model = inception_v4(in_channels=3, num_classes=1000, dropout=0.7)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)