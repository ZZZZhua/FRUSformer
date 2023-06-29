import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class Block(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_s=1,padd=1,str=1,gro=2,bn_mom=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim_in,dim_out,kernel_size=kernel_s,padding=padd,stride=1,groups=gro),
            nn.BatchNorm2d(dim_out,momentum=bn_mom),
            nn.Conv2d(dim_out, dim_in, kernel_size=kernel_s, padding=padd, stride=str, groups=gro),
            nn.ReLU()
        )
    def forward(self,x):
        return self.block(x)

class GLM(nn.Module):
    def __init__(self,dimin,bn_mom=0.1,padw=2,kernel=5,Num=3):
        super(GLM, self).__init__()
        """
        param dimin:  输入的维度
        """
        #获取特征1
        self.branch1 = nn.ModuleList(
            [
                Block(
                    dimin,dimin*4,kernel_s=3,padd=1,str=1,gro=2
                )
                for _ in range(Num)
            ]

        )
        self.branch2 = nn.ModuleList(
            [
                Block(
                    dimin,dimin*4,kernel_s=1,padd=0,str=1,gro=2
                )
                for _ in range(Num)
            ]

        )
        self.branch3 = nn.ModuleList(
            [
                Block(
                    dimin,dimin*4,kernel_s=5,padd=padw,str=1,gro=2
                )
                for _ in range(Num)
            ]

        )

        self.downsaple = nn.Sequential(
            nn.Conv2d(dimin*3,dimin*3,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(dimin*3, momentum=bn_mom),
            nn.ReLU()
        )

    def forward(self,x):
        b,c,h,w = x.size()
        res = x
        bl1 = x
        bl2 = x
        bl3 = x
        for i,blk in enumerate(self.branch1):
            bl1 = blk.forward(bl1)
        bl1 = bl1 +x
        for i, blk in enumerate(self.branch2):
            bl2 = blk.forward(bl2)
        bl2 = bl2 + x
        for i, blk in enumerate(self.branch3):
            bl3 = blk.forward(bl3)
        bl3 = bl3 + x
        res = torch.cat([bl1, bl2, bl3],dim=1)
        res = self.downsaple(res)
        return res
