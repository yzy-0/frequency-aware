import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

class PHM(nn.Module):
    """
    The implementation of RGB-induced details enhancement module.
    """

    def __init__(self, channel: int, k=3) -> None:
        super(PHM, self).__init__()

        self.Avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            DSConv3x3(channel, channel),
            nn.Sigmoid()
        )
        self.CBR = nn.Sequential(
            DSConv3x3(channel,channel,),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv1x3 = nn.Conv2d(channel// 2 , channel// 2 , kernel_size=(1, k),stride=2, padding=(0, k // 2))
        self.conv3x1 = nn.Conv2d(channel// 2 , channel // 2, kernel_size=(k, 1), padding=(k // 2, 0))

        self.conv3x1_2 = nn.Conv2d(channel // 2, channel // 2, kernel_size=(k, 1), stride=2, padding=(k // 2, 0))
        self.conv1x3_2 = nn.Conv2d(channel // 2, channel // 2 , kernel_size=(1, k), padding=(0, k // 2))

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, input_enhance: Tensor, input_rgb: Tensor) -> Tensor:
        rgbd1x3 = self.conv1x3(input_enhance)
        rgbd3x1 = self.conv3x1(rgbd1x3)

        rgbdd3x1 = self.conv3x1_2(input_enhance)
        rgbdd1x3 = self.conv1x3_2(rgbdd3x1)
        rgbd1331 = torch.cat([rgbdd1x3, rgbd3x1], dim=1)
        rgbd1331 = self.CBR(rgbd1331)

        x, _ = torch.max(input_rgb, dim=1, keepdim=True)
        x = self.conv2(self.conv1(x))
        mask_MC = torch.sigmoid(x)  ###1xHxW

        x_globalAve = self.Avg(input_rgb)  ###1xHxW

        fuse = rgbd1331 * mask_MC + rgbd1331 * x_globalAve
        rgb_enhance = fuse + input_rgb

        return rgb_enhance