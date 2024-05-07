import torch
import torch.nn as nn


__all__ = ['InceptionV3', 'inception_v3']


class InceptionV3(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(InceptionV3, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(in_dims=in_channels, out_dims=32, kernel_size=3, stride=2, padding=0)
        self.Conv2d_2a_3x3 = BasicConv2d(in_dims=32, out_dims=32, kernel_size=3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(in_dims=32, out_dims=64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(in_dims=64, out_dims=80, kernel_size=1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(in_dims=80, out_dims=192, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(in_dims=192, pool_features=32)
        self.Mixed_5c = InceptionA(in_dims=256, pool_features=64)
        self.Mixed_5d = InceptionA(in_dims=288, pool_features=64)

        self.Mixed_6a = InceptionB(in_dims=288)

        self.Mixed_6b = InceptionC(in_dims=768, ch7x7=128)
        self.Mixed_6c = InceptionC(in_dims=768, ch7x7=160)
        self.Mixed_6d = InceptionC(in_dims=768, ch7x7=160)
        self.Mixed_6e = InceptionC(in_dims=768, ch7x7=192)

        self.Mixed_7a = InceptionD(in_dims=768)
        self.Mixed_7b = InceptionE(in_dims=1280)
        self.Mixed_7c = InceptionE(in_dims=2048)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x):
        out = self.Conv2d_1a_3x3(x)
        out = self.Conv2d_2a_3x3(out)
        out = self.Conv2d_2b_3x3(out)
        out = self.maxpool1(out)
        out = self.Conv2d_3b_1x1(out)
        out = self.Conv2d_4a_3x3(out)
        out = self.maxpool2(out)

        out = self.Mixed_5b(out)
        out = self.Mixed_5c(out)
        out = self.Mixed_5d(out)

        out = self.Mixed_6a(out)

        out = self.Mixed_6b(out)
        out = self.Mixed_6c(out)
        out = self.Mixed_6d(out)
        out = self.Mixed_6e(out)

        out = self.Mixed_7a(out)

        out = self.Mixed_7b(out)
        out = self.Mixed_7c(out)

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


class InceptionA(nn.Module):
    # Inception module in Figure 5
    def __init__(self, in_dims, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=64, kernel_size=1, stride=1, padding=0)

        self.branch5x5_1 = BasicConv2d(in_dims=in_dims, out_dims=48, kernel_size=1, stride=1, padding=0)
        self.branch5x5_2 = BasicConv2d(in_dims=48, out_dims=64, kernel_size=5, stride=1, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_dims=in_dims, out_dims=64, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(in_dims=96, out_dims=96, kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_dims=in_dims, out_dims=pool_features, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_branch1x1 = self.branch1x1(x)

        out_branch5x5 = self.branch5x5_1(x)
        out_branch5x5 = self.branch5x5_2(out_branch5x5)

        out_branch3x3dbl = self.branch3x3dbl_1(x)
        out_branch3x3dbl = self.branch3x3dbl_2(out_branch3x3dbl)
        out_branch3x3dbl = self.branch3x3dbl_3(out_branch3x3dbl)

        out_branch_pool = self.avgpool(x)
        out_branch_pool = self.branch_pool(out_branch_pool)

        out = torch.cat([out_branch1x1, out_branch5x5, out_branch3x3dbl, out_branch_pool], dim=1)

        return out


class InceptionB(nn.Module):
    # Grid size reduction block between the Inception modules in Figures 5 and 6
    def __init__(self, in_dims):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_dims=in_dims, out_dims=384, kernel_size=3, stride=2, padding=0)

        self.branch3x3dbl_1 = BasicConv2d(in_dims=in_dims, out_dims=64, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_dims=64, out_dims=96, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(in_dims=96, out_dims=96, kernel_size=3, stride=2, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out_branch3x3 = self.branch3x3(x)

        out_branch3x3dbl = self.branch3x3dbl_1(x)
        out_branch3x3dbl = self.branch3x3dbl_2(out_branch3x3dbl)
        out_branch3x3dbl = self.branch3x3dbl_3(out_branch3x3dbl)

        out_branch_pool = self.maxpool(x)

        out = torch.cat([out_branch3x3, out_branch3x3dbl, out_branch_pool], dim=1)

        return out


class InceptionC(nn.Module):
    # Inception module in Figure 6
    def __init__(self, in_dims, ch7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_dims=in_dims, out_dims=ch7x7, kernel_size=1, stride=1, padding=0)
        self.branch7x7_2 = BasicConv2d(in_dims=ch7x7, out_dims=ch7x7, kernel_size=(1,7), stride=1, padding=(0,3))
        self.branch7x7_3 = BasicConv2d(in_dims=ch7x7, out_dims=192, kernel_size=(7,1), stride=1, padding=(3,0))

        self.branch7x7dbl_1 = BasicConv2d(in_dims=in_dims, out_dims=ch7x7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(in_dims=ch7x7, out_dims=ch7x7, kernel_size=(7,1), stride=1, padding=(3,0))
        self.branch7x7dbl_3 = BasicConv2d(in_dims=ch7x7, out_dims=ch7x7, kernel_size=(1,7), stride=1, padding=(0,3))
        self.branch7x7dbl_4 = BasicConv2d(in_dims=ch7x7, out_dims=ch7x7, kernel_size=(7,1), stride=1, padding=(3,0))
        self.branch7x7dbl_5 = BasicConv2d(in_dims=ch7x7, out_dims=192, kernel_size=(1,7), stride=1, padding=(0,3))

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out_branch1x1 = self.branch1x1(x)

        out_branch7x7 = self.branch7x7_1(x)
        out_branch7x7 = self.branch7x7_2(out_branch7x7)
        out_branch7x7 = self.branch7x7_3(out_branch7x7)

        out_branch7x7dbl = self.branch7x7dbl_1(x)
        out_branch7x7dbl = self.branch7x7dbl_2(out_branch7x7dbl)
        out_branch7x7dbl = self.branch7x7dbl_3(out_branch7x7dbl)
        out_branch7x7dbl = self.branch7x7dbl_4(out_branch7x7dbl)
        out_branch7x7dbl = self.branch7x7dbl_5(out_branch7x7dbl)

        out_branch_pool = self.avgpool(x)
        out_branch_pool = self.branch_pool(out_branch_pool)

        out = torch.cat([out_branch1x1, out_branch7x7, out_branch7x7dbl, out_branch_pool], dim=1)

        return out


class InceptionD(nn.Module):
    # Grid size reduction block between the Inception modules in Figures 6 and 7
    def __init__(self, in_dims):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2 = BasicConv2d(in_dims=192, out_dims=320, kernel_size=3, stride=2, padding=0)

        self.branch7x7x3_1 = BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1, stride=1, padding=0)
        self.branch7x7x3_2 = BasicConv2d(in_dims=192, out_dims=192, kernel_size=(1,7), stride=1, padding=(0,3))
        self.branch7x7x3_3 = BasicConv2d(in_dims=192, out_dims=192, kernel_size=(7,1), stride=1, padding=(3,0))
        self.branch7x7x3_4 = BasicConv2d(in_dims=192, out_dims=192, kernel_size=3, stride=2, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out_branch3x3 = self.branch3x3_1(x)
        out_branch3x3 = self.branch3x3_2(out_branch3x3)

        out_branch7x7x3 = self.branch7x7x3_1(x)
        out_branch7x7x3 = self.branch7x7x3_2(out_branch7x7x3)
        out_branch7x7x3 = self.branch7x7x3_3(out_branch7x7x3)
        out_branch7x7x3 = self.branch7x7x3_4(out_branch7x7x3)

        out_branch_pool = self.maxpool(x)

        out = torch.cat([out_branch3x3, out_branch7x7x3, out_branch_pool], dim=1)

        return out


class InceptionE(nn.Module):
    # Inception module in Figure 7
    def __init__(self, in_dims):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_dims=in_dims, out_dims=320, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = BasicConv2d(in_dims=in_dims, out_dims=384, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2a = BasicConv2d(in_dims=384, out_dims=384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch3x3_2b = BasicConv2d(in_dims=384, out_dims=384, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3x3dbl_1 = BasicConv2d(in_dims=in_dims, out_dims=448, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(in_dims=448, out_dims=384, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(in_dims=384, out_dims=384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch3x3dbl_3b = BasicConv2d(in_dims=384, out_dims=384, kernel_size=(3,1), stride=1, padding=(1,0))

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_dims=in_dims, out_dims=192, kernel_size=1)

    def forward(self, x):
        out_branch1x1 = self.branch1x1(x)

        out_branch3x3 = self.branch3x3_1(x)
        out_branch3x3 = torch.cat([self.branch3x3_2a(out_branch3x3), self.branch3x3_2b(out_branch3x3)], dim=1)

        out_branch3x3dbl = self.branch3x3dbl_1(x)
        out_branch3x3dbl = self.branch3x3dbl_2(out_branch3x3dbl)
        out_branch3x3dbl = torch.cat([self.branch3x3dbl_3a(out_branch3x3dbl), self.branch3x3dbl_3b(out_branch3x3dbl)], dim=1)

        out_branch_pool = self.avgpool(x)
        out_branch_pool = self.branch_pool(out_branch_pool)

        out = torch.cat([out_branch1x1, out_branch3x3, out_branch3x3dbl, out_branch_pool], dim=1)

        return out


def inception_v3(**kwargs):
    model = InceptionV3(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 224
    model = inception_v3(in_channels=3, num_classes=1000, dropout=0.7)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)