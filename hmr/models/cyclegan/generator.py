
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv, deconv, conv3x3, conv3x3_bn_relu
from .discriminator import ResNet


class UPerNetDecoder(nn.Module):
    def __init__(self, num_class=1, fc_dim=256,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(16, 32, 64, 128), fpn_dim=256):
        super().__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(f,
                              size=conv_x.size()[2:],
                              mode='bilinear',
                              align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class UPerNet(nn.Module):
    def __init__(self, conv_dim):
        super().__init__()
        self.encoder = ResNet(conv_dim)
        self.decoder = UPerNetDecoder(fc_dim=conv_dim, pool_scales=(1, 2, 3, 6),
                                      fpn_inplanes=(conv_dim // 8, conv_dim // 4, conv_dim // 2, conv_dim), fpn_dim=conv_dim)

    def forward(self, x):
        out = self.encoder(x, output_cls=False)
        out = self.decoder(out)
        out = F.interpolate(out, (224, 224),
                            mode='bilinear',
                            align_corners=True)
        return out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, conv_dim, n_class=1):
        super().__init__()

        self.dconv_down1 = double_conv(1, conv_dim // 8)
        self.dconv_down2 = double_conv(conv_dim // 8, conv_dim // 4)
        self.dconv_down3 = double_conv(conv_dim // 4, conv_dim // 2)
        self.dconv_down4 = double_conv(conv_dim // 2, conv_dim)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = lambda x: F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(
            conv_dim // 2 + conv_dim, conv_dim // 2)
        self.dconv_up2 = double_conv(
            conv_dim // 4 + conv_dim // 2, conv_dim // 4)
        self.dconv_up1 = double_conv(
            conv_dim // 4 + conv_dim // 8, conv_dim // 8)

        self.conv_last = nn.Conv2d(conv_dim // 8, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
