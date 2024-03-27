import torch
import torch.nn as nn

__all__ = ['VGGNet', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


vgg_types = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNet(nn.Module):
    def __init__(self, model='vgg13', in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.features = self.make_layers(cfgs=vgg_types[model], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def make_layers(self, cfgs, batch_norm=True):
        layers = []
        in_channels = self.in_channels
        for cfg in cfgs:
            if type(cfg) == int:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(num_features=cfg), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.ReLU(inplace=True)]
                in_channels = cfg
            elif cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def vgg11(**kwargs):
    model = VGGNet(model='vgg11', **kwargs)

    return model


def vgg13(**kwargs):
    model = VGGNet(model='vgg13', **kwargs)

    return model


def vgg16(**kwargs):
    model = VGGNet(model='vgg16', **kwargs)

    return model


def vgg19(**kwargs):
    model = VGGNet(model='vgg19', **kwargs)

    return model


if __name__ == '__main__':
    img_size = 224
    model = VGGNet(model='vgg13', in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)