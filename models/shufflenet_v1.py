import torch
import torch.nn as nn


__all__ = ['ShuffleNetV1', 'shufflenet_v1']


def shuffle_channel(x, num_groups):
    B, C, H, W = x.size()
    C_per_groups = int(C//num_groups)

    x = x.view(B, num_groups, C_per_groups, H, W)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(B, C, H, W)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_dims, out_dims, num_groups, combine_method='add'):
        super(ShuffleUnit, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.inner_dims = int(self.out_dims//4)

        self.combine_method = combine_method
        if self.combine_method == 'add':
            self.stride = 1
            self.residual = nn.Sequential()
            self.combine = self._add
        elif self.combine_method == 'concat':
            self.stride = 2
            self.residual = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.combine = self._concat
            self.out_dims -= self.in_dims

        self.gconv1x1a = nn.Sequential(
            nn.Conv2d(self.in_dims, self.inner_dims, kernel_size=1, stride=1, padding=0, groups=num_groups, bias=False),
            nn.BatchNorm2d(num_features=self.inner_dims),
            nn.ReLU(inplace=True)
        )
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(self.inner_dims, self.inner_dims, kernel_size=3, stride=self.stride, padding=1, groups=self.inner_dims, bias=False),
            nn.BatchNorm2d(num_features=self.inner_dims),
            nn.ReLU(inplace=True)
        )
        self.gconv1x1b = nn.Sequential(
            nn.Conv2d(self.inner_dims, self.out_dims, kernel_size=1, stride=1, padding=0, groups=num_groups, bias=False),
            nn.BatchNorm2d(num_features=self.out_dims)
        )
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), dim=1)

    def forward(self, x):
        out_residual = self.residual(x)

        out_shortcut = self.gconv1x1a(x)
        out_shortcut = self.dwconv3x3(out_shortcut)
        out_shortcut = self.gconv1x1b(out_shortcut)

        out = self.combine(out_residual, out_shortcut)
        out = self.relu(out)

        return out


class ShuffleNetV1(nn.Module):
    def __init__(self, in_channels, num_classes, num_groups=3):
        super(ShuffleNetV1, self).__init__()
        self.num_groups = num_groups
        if self.num_groups == 1:
            self.out_dims = [24, 144, 288, 576]
        elif self.num_groups == 2:
            self.out_dims = [24, 200, 400, 800]
        elif self.num_groups == 3:
            self.out_dims = [24, 240, 480, 960]
        elif self.num_groups == 4:
            self.out_dims = [24, 272, 544, 1088]
        elif self.num_groups == 8:
            self.out_dims = [24, 384, 768, 1536]
        else:
            raise ValueError(
                '{} groups is not supported for 1x1 Grouped Convolutions'.format(num_groups)
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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(self.out_dims[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_stages(self, idx_stage):
        nstages = [3,7,3]
        layers = []
        layers.append(ShuffleUnit(in_dims=self.out_dims[idx_stage-1], out_dims=self.out_dims[idx_stage], num_groups=1, combine_method='concat'))
        for _ in range(nstages[idx_stage-1]):
            if idx_stage == 1:
                layers.append(ShuffleUnit(in_dims=self.out_dims[idx_stage], out_dims=self.out_dims[idx_stage], num_groups=1, combine_method='add'))
            else:
                layers.append(ShuffleUnit(in_dims=self.out_dims[idx_stage], out_dims=self.out_dims[idx_stage], num_groups=self.num_groups, combine_method='add'))

        return nn.Sequential(*layers)


def shufflenet_v1(**kwargs):
    return ShuffleNetV1(**kwargs)


if __name__ == '__main__':
    img_size = 224

    model = shufflenet_v1(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)

    output = model(input)
    print(output.shape)