
import torch
import torch.nn as nn
#1*32*64*64
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out,pad=1,kernel=3):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel, padding=pad, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class localConv2d(nn.Module):
    def  __init__(self, in_channels, band_kernel_size=5):
        super(localConv2d, self).__init__()
        self.Down = nn.Sequential(

        )

        self.branch1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1)

        self.branch3x3 =nn.Sequential(
            nn.Conv2d(in_channels,in_channels*3,kernel_size=1),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1),
            nn.Conv2d(in_channels*3,in_channels,kernel_size=3, padding=1)
        )

        self.branchmax_pooling =nn.Sequential(
            nn.MaxPool2d(kernel_size=3,padding=1,stride= 1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.branch5x5=nn.Sequential(
            nn.Conv2d(in_channels,in_channels*3,kernel_size=1),
            depthwise_separable_conv(in_channels*3,in_channels,pad=2,kernel=band_kernel_size)
        )

    def forward(self,x):
        b1=self.branch1x1(x)
        b3=self.branch3x3(x)
        b5=self.branch5x5(x)
        bmax=self.branchmax_pooling(x)
        output = torch.cat([b1,b3,b5,bmax],dim=1)

        return output

