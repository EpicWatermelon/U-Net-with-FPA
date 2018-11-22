import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.lhs_1x = nn.Sequential(ConvBNReLU(1, 16), ConvBNReLU(16, 16))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(16, 32), ConvBNReLU(32, 32))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(64, 128), ConvBNReLU(128, 128))

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(128, 128), ConvBNReLU(128, 128))
        self.fpa = FeaturePyramidAttention(128, 128)

        self.rhs_8x = UpConcat(128, 128)
        self.rhs_4x = UpConcat(128, 64)
        self.rhs_2x = UpConcat(64, 32)
        self.rhs_1x = UpConcat(32, 16)

        self.classify = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, inputs):
        lhs_1x = self.lhs_1x(inputs)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)

        bottom = self.bottom(lhs_8x)
        bottom = self.fpa(bottom)

        rhs_8x = self.rhs_8x(lhs_8x, bottom)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)

        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)
        outputs = self.classify(rhs_1x)

        return outputs


class FeaturePyramidAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidAttention, self).__init__()
        self.conv7 = nn.Sequential(nn.MaxPool2d(2, 2), ConvBNReLU(in_channels, out_channels, 7, 1, 3),
                                   )
        self.conv5 = nn.Sequential(nn.MaxPool2d(2, 2), ConvBNReLU(in_channels, out_channels, 5, 1, 2),
                                   )
        self.conv3 = nn.Sequential(nn.MaxPool2d(2, 2), ConvBNReLU(in_channels, out_channels, 3, 1, 1),
                                   )
        self.conv7_2 = ConvBNReLU(in_channels, out_channels, 7, 1, 3)
        self.conv5_2 = ConvBNReLU(in_channels, out_channels, 5, 1, 2)
        self.conv3_2 = ConvBNReLU(in_channels, out_channels, 3, 1, 1)
        self.conv_global = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        u = self.conv_global(x)

        d1 = self.conv7(x)
        d2 = self.conv5(d1)
        d3 = self.conv3(d2)

        d1_1 = self.conv7_2(d1)
        d2_1 = self.conv5_2(d2)
        d3_1 = self.conv3_2(d3)

        d3_1 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(d3_1)
        d2_1 = d2_1 + d3_1
        d2_1 = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(d2_1)
        d1_1 = d1_1 + d2_1
        d1_1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(d1_1)

        m = torch.mul(x, d1_1)
        a1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(u)
        final = m + a1
        return final


def ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True)
                         )


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConcat, self).__init__()
        # Right hand side needs `Upsample`
        self.rhs_up = nn.Upsample(scale_factor=2)
        self.conv_fit = ConvBNReLU(in_channels + out_channels, out_channels)
        self.conv = nn.Sequential(ConvBNReLU(out_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, lhs, rhs):
        rhs = self.rhs_up(rhs)
        rhs = make_same(lhs, rhs)
        cat = torch.cat((lhs, rhs), dim=1)
        return self.conv(self.conv_fit(cat))


def make_same(good, evil):
    """
    good / evil could be 1-d, 2-d or 3-d Tensor, i.e., [batch_size, channels, (depth,) (height,) width]
    Implemented by tensor.narrow
    """
    # Make evil bigger
    g, e = good.size(), evil.size()
    ndim = len(e) - 2
    pad = int(max(np.subtract(g, e)))
    if pad > 0:
        pad = tuple([pad] * ndim * 2)
        evil = F.pad(evil, pad, mode='replicate')

    # evil > good:
    e = evil.size()  # update
    for i in range(2, len(e)):
        diff = (e[i] - g[i]) // 2
        evil = evil.narrow(i, diff, g[i])
    return evil

