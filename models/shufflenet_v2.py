import torch
import torch.nn as nn


__all__ = ['ShuffleNetV2', 'shufflenet_v2']


def shuffle_channel(x, num_groups):
    B, C, H, W = x.size()
    C_per_groups = int(C//num_groups)

    x = x.view(B, num_groups, C_per_groups, H, W)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(B, C, H, W)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_dims, out_dims, downsample=False):
        super(ShuffleUnit, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.inner_dims = int(out_dims//2)

        self.downsample = downsample
        if self.downsample:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_dims, self.in_dims, kernel_size=3, stride=2, padding=1, groups=self.in_dims, bias=False),
                nn.BatchNorm2d(num_features=self.in_dims),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.in_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(self.in_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=3, stride=2, padding=1, groups=self.inner_dims, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),

                nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=3, stride=1, padding=1, groups=self.inner_dims, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),

                nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.inner_dims),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = None
        if self.downsample:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)

        out = shuffle_channel(out, num_groups=2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, net_type=1):
        super(ShuffleNetV2, self).__init__()
        self.net_type = net_type
        if self.net_type == 0.5:
            self.out_dims = [24, 48, 96, 192, 1024]
        elif self.net_type == 1:
            self.out_dims = [24, 116, 232, 464, 1024]
        elif self.net_type == 1.5:
            self.out_dims = [24, 176, 352, 704, 1024]
        elif self.net_type == 2:
            self.out_dims = [24, 244, 488, 976, 2048]
        else:
            raise ValueError(
                '{} type is not supported'.format(net_type)
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.out_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stages(idx_stage=1)
        self.stage3 = self._make_stages(idx_stage=2)
        self.stage4 = self._make_stages(idx_stage=3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.out_dims[-2], self.out_dims[-1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.out_dims[-1]),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(self.out_dims[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_stages(self, idx_stage):
        nstages = [3,7,3]
        layers = []
        layers.append(ShuffleUnit(in_dims=self.out_dims[idx_stage-1], out_dims=self.out_dims[idx_stage], downsample=True))
        for _ in range(nstages[idx_stage-1]):
            layers.append(ShuffleUnit(in_dims=self.out_dims[idx_stage], out_dims=self.out_dims[idx_stage], downsample=False))

        return nn.Sequential(*layers)


def shufflenet_v2(**kwargs):
    return ShuffleNetV2(**kwargs)


if __name__ == '__main__':
    img_size = 224

    model = shufflenet_v2(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)

    output = model(input)
    print(output.shape)