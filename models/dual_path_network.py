import torch
import torch.nn as nn


__all__ = ['DualPathNetwork', 'dpn92', 'dpn98']


class PreActConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, **kwargs):
        super(PreActConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, out_dims, **kwargs)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class DualPathBlock(nn.Module):
    def __init__(self, in_dims, res_dims_1x1a, res_dims_3x3b, res_dims_1x1c, inc_dims, num_groups, block_type='normal'):
        super(DualPathBlock, self).__init__()
        self.res_dims_1x1c = res_dims_1x1c

        if block_type == 'proj':
            self.stride = 1
            self.is_proj = True
        elif block_type == 'down':
            self.stride = 2
            self.is_proj = True
        elif block_type == 'normal':
            self.stride = 1
            self.is_proj = False

        if self.is_proj:
            self.c1x1_w = PreActConvBlock(in_dims=in_dims, out_dims=res_dims_1x1c+2*inc_dims,
                                          kernel_size=1, stride=self.stride, padding=0, bias=False)

        self.layers = nn.Sequential(
            PreActConvBlock(in_dims=in_dims, out_dims=res_dims_1x1a, kernel_size=1, stride=1, padding=0, bias=False),
            PreActConvBlock(in_dims=res_dims_1x1a, out_dims=res_dims_3x3b, kernel_size=3, stride=self.stride, padding=1, groups=num_groups, bias=False),
            PreActConvBlock(in_dims=res_dims_3x3b, out_dims=res_dims_1x1c+inc_dims, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, list) else x

        if self.is_proj:
            x_s = self.c1x1_w(x_in)
            x_s1 = x_s[:, :self.res_dims_1x1c, :, :]
            x_s2 = x_s[:, self.res_dims_1x1c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]

        x_in = self.layers(x_in)

        out_res = x_s1 + x_in[:, :self.res_dims_1x1c, :, :]
        out_dense = torch.cat([x_s2, x_in[:, self.res_dims_1x1c:, :]], dim=1)

        return [out_res, out_dense]


class DualPathNetwork(nn.Module):
    def __init__(self, in_channels, num_classes, init_dims=64, inner_dims=96, num_groups=32, num_blocks=[3,4,20,3], inc_dims=[16,32,24,128]):
        super(DualPathNetwork, self).__init__()
        self.features = nn.Sequential()

        # conv1
        self.features.add_module(
            'conv_1',
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=init_dims, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_features=init_dims),
                nn.ReLU(inplace=True)
            )
        )
        self.features.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # conv2
        bw = 256
        inc = inc_dims[0]
        R = int((inner_dims*bw)/256)
        self.features.add_module(
            'conv_2_0',
            DualPathBlock(init_dims, R, R, bw, inc, num_groups, 'proj')
        )
        inner_dims = bw + 3 * inc
        for i in range(1, num_blocks[0]):
            self.features.add_module(
                'conv_2_{}'.format(i),
                DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'normal')
            )
            inner_dims += inc

        # conv3
        bw = 512
        inc = inc_dims[1]
        R = int((inner_dims*bw)/256)
        self.features.add_module(
            'conv_3_0',
            DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'down')
        )
        inner_dims = bw + 3 * inc
        for i in range(1, num_blocks[1]):
            self.features.add_module(
                'conv_3_{}'.format(i),
                DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'normal')
            )
            inner_dims += inc

        # conv4
        bw = 1024
        inc = inc_dims[2]
        R = int((inner_dims*bw)/256)
        self.features.add_module(
            'conv_4_0',
            DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'down')
        )
        inner_dims = bw + 3 * inc
        for i in range(1, num_blocks[2]):
            self.features.add_module(
                'conv_4_{}'.format(i),
                DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'normal')
            )
            inner_dims += inc

        # conv5
        bw = 2048
        inc = inc_dims[2]
        R = int((inner_dims*bw)/256)
        self.features.add_module(
            'conv_5_0',
            DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'down')
        )
        inner_dims = bw + 3 * inc
        for i in range(1, num_blocks[3]):
            self.features.add_module(
                'conv_5_{}'.format(i),
                DualPathBlock(inner_dims, R, R, bw, inc, num_groups, 'normal')
            )
            inner_dims += inc

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(inner_dims, num_classes)

    def forward(self, x):
        out = torch.cat(self.features(x), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


def dpn92(**kwargs):
    return DualPathNetwork(init_dims=64, inner_dims=96, num_groups=32, num_blocks=[3,4,20,3], inc_dims=[16,32,24,128], **kwargs)


def dpn98(**kwargs):
    return DualPathNetwork(init_dims=96, inner_dims=160, num_groups=40, num_blocks=[3,6,20,3], inc_dims=[16,32,32,128], **kwargs)


if __name__ == '__main__':
    img_size = 224
    model = dpn92(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)