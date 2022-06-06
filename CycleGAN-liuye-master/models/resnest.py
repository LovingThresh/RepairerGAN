import torch.nn as nn
import torchsummary


class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


# 明确组卷积(Group Convolution)的作用
class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_channels, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_channels = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_channels, self.group_channels, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_channels, self.group_channels, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_channels, self.group_channels * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_channels * 2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, self.group_channels * 2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_channels * 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out = out + self.short_cut(x)
        return nn.ReLU()(out)


#
class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: list, cardinality, group_depth, num_classes):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self._make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self._make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self._make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self._make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(self.channels, num_classes)   # 224x224 input size

    def _make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.MaxPool2d((3, 3), 2, 1)(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = nn.AvgPool2d(7)(out)
        out = out.view(out.size(0), -1)
        out = nn.Softmax(self.fc(out))
        return out


def resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


model = resNeXt50_32x4d(num_classes=1000)
torchsummary.summary(model, input_size=(3, 224, 224), device='cpu')
