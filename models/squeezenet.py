import torch
import torch.nn as nn

__all__ = ['squeezenet']

class FireModule(nn.Module):
    def __init__(self, in_dims, squeeze_dims, expand11_dims, expand33_dims):
        super(FireModule, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_dims, squeeze_dims, kernel_size=1),
            nn.BatchNorm2d(num_features=squeeze_dims),
            nn.ReLU(inplace=True),
        )
        self.expand11 = nn.Sequential(
            nn.Conv2d(squeeze_dims, expand11_dims, kernel_size=1),
            nn.BatchNorm2d(num_features=expand11_dims),
            nn.ReLU(inplace=True),
        )
        self.expand33 = nn.Sequential(
            nn.Conv2d(squeeze_dims, expand33_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=expand33_dims),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x11 = self.expand11(x)
        x33 = self.expand33(x)
        x = torch.cat([x11,x33], dim=1)
        x = self.relu(x)

        return x


class SqueezeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, p_dropout=0.5):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)

        return x


def squeezenet(**kwargs):
    model = SqueezeNet(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 224

    model = squeezenet(in_channels=3, num_classes=1000, p_dropout=0.5)

    input = torch.randn(1, 3, img_size, img_size)

    output = model(input)
    print(output.shape)