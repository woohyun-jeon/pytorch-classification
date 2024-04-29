import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResidualAttentionNetwork', 'ran56', 'ran92']


# set Residual Attention Network configuration
cfgs = [64, 256, 512, 1024, 2048]


class PreActResUnit(nn.Module):
    def __init__(self, in_dims, out_dims, stride=1):
        super(PreActResUnit, self).__init__()
        self.residual = nn.Sequential(
            # 1x1 conv layer
            nn.BatchNorm2d(num_features=in_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims, int(out_dims//4), kernel_size=1, stride=1, padding=0, bias=False),

            # 3x3 conv layer
            nn.BatchNorm2d(num_features=int(out_dims//4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_dims//4), int(out_dims//4), kernel_size=3, stride=stride, padding=1, bias=False),

            # 1x1 conv layer
            nn.BatchNorm2d(num_features=int(out_dims//4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_dims//4), out_dims, kernel_size=1, stride=1, padding=0, bias=False),
        )

        if stride != 1 or in_dims != out_dims:
            self.shortcut = nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)

        return out


class AttentionModule(nn.Module):
    def __init__(self, in_dims, out_dims, p=1, t=2, r=1):
        super(AttentionModule, self).__init__()
        self.first_layer = self._make_layers(in_dims, out_dims, num_layers=p)

        self.trunk_branch = self._make_layers(in_dims, out_dims, num_layers=t)

        self.softmask_branch_down1 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_down2 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_down3 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_bridge = nn.Sequential(
            self._make_layers(in_dims, out_dims, num_layers=r),
            self._make_layers(in_dims, out_dims, num_layers=r)
        )
        self.softmask_branch_up1 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_up2 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_up3 = self._make_layers(in_dims, out_dims, num_layers=r)
        self.softmask_branch_shortcut1 = PreActResUnit(in_dims, out_dims, stride=1)
        self.softmask_branch_shortcut2 = PreActResUnit(in_dims, out_dims, stride=1)
        self.softmask_branch_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmask_branch_out = nn.Sequential(
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.final_layer = self._make_layers(in_dims, out_dims, num_layers=p)

    def forward(self, x):
        # first layer
        x = self.first_layer(x)
        size1 = (x.size(2), x.size(3))

        # trunk branch
        out_trunk = self.trunk_branch(x)

        # softmask branch first down part
        out_softmask_resunit1 = self.softmask_branch_maxpool(x)
        out_softmask_resunit1 = self.softmask_branch_down1(out_softmask_resunit1)

        size2 = (out_softmask_resunit1.size(2), out_softmask_resunit1.size(3))
        out_softmask_shortcut1 = self.softmask_branch_shortcut1(out_softmask_resunit1)

        # softmask branch second down part
        out_softmask_resunit2 = self.softmask_branch_maxpool(out_softmask_resunit1)
        out_softmask_resunit2 = self.softmask_branch_down2(out_softmask_resunit2)

        size3 = (out_softmask_resunit2.size(2), out_softmask_resunit2.size(3))
        out_softmask_shortcut2 = self.softmask_branch_shortcut2(out_softmask_resunit2)

        # softmask branch third down part
        out_softmask_resunit3 = self.softmask_branch_maxpool(out_softmask_resunit2)
        out_softmask_resunit3 = self.softmask_branch_down3(out_softmask_resunit3)

        out_softmask_bridge = self.softmask_branch_bridge(out_softmask_resunit3)

        # softmask branch first up part
        out_softmask_resunit4 = self.softmask_branch_up1(out_softmask_bridge)
        out_softmask_resunit4 = F.interpolate(out_softmask_resunit4, size=size3, mode='bilinear', align_corners=True)
        out_softmask_resunit4 += out_softmask_shortcut2

        # softmask branch second up part
        out_softmask_resunit5 = self.softmask_branch_up2(out_softmask_resunit4)
        out_softmask_resunit5 = F.interpolate(out_softmask_resunit5, size=size2, mode='bilinear', align_corners=True)
        out_softmask_resunit5 += out_softmask_shortcut1

        # softmask branch thrid up part
        out_softmask_resunit6 = self.softmask_branch_up3(out_softmask_resunit5)
        out_softmask_resunit6 = F.interpolate(out_softmask_resunit6, size=size1, mode='bilinear', align_corners=True)

        # last layer
        out_softmask = self.softmask_branch_out(out_softmask_resunit6)
        out = (1 + out_softmask) * out_trunk
        out = self.final_layer(out)

        return out

    def _make_layers(self, in_dims, out_dims, num_layers=2):
        layers = []
        for _ in range(num_layers):
            layers.append(PreActResUnit(in_dims, out_dims, stride=1))

        return nn.Sequential(*layers)


class ResidualAttentionNetwork(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks=[1,1,1]):
        super(ResidualAttentionNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cfgs[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=cfgs[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stages(cfgs[0], cfgs[1], num_blocks[0], block=AttentionModule, stride=1)
        self.stage2 = self._make_stages(cfgs[1], cfgs[2], num_blocks[1], block=AttentionModule, stride=2)
        self.stage3 = self._make_stages(cfgs[2], cfgs[3], num_blocks[2], block=AttentionModule, stride=2)
        self.stage4 = nn.Sequential(
            PreActResUnit(cfgs[3], cfgs[4], stride=1),
            PreActResUnit(cfgs[4], cfgs[4], stride=1),
            PreActResUnit(cfgs[4], cfgs[4], stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(cfgs[4], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_stages(self, in_dims, out_dims, num_blocks, block, stride):
        stage = []
        stage.append(PreActResUnit(in_dims, out_dims, stride))

        for _ in range(num_blocks):
            stage.append(block(out_dims, out_dims))

        return nn.Sequential(*stage)


def ran56(**kwargs):
    return ResidualAttentionNetwork(num_blocks=[1,1,1], **kwargs)


def ran92(**kwargs):
    return ResidualAttentionNetwork(num_blocks=[1,2,3], **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = ran56(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)