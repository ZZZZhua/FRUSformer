
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        """

        :param channel:   输入通道数
        :param reduction:   特征图通道降低的倍数
        """
        #提取全局特征
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #学习通道之间的相关性
        #使用卷积的方式来获得 通道数先减少在会变成原来的  这里是获取了通道之间的相关性
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )


    def forward(self, x):
        #batch channel height, width
        b, c, h, w = x.size()
        #b ,c ,h, w -> b c 1 1 -> bc
        y = self.avg_pool(x).view(b, c)
        #将其转化为原格式并且学习参数  b,c ->b, c ,1 ,1
        y = y.view(b, c, 1, 1)
        y = self.fc1(y)
        pro = x * y
        return pro
