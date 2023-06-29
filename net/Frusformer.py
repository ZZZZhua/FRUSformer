
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat

from torchsummary import summary
from nets.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from nets.GlocalM import GLM,LocalM
from nets.crossAttention import localADeepMerge
from net.localM import localConv2d

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    
    
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class FursformerHead(nn.Module):
   
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(FursformerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels*3, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels*3, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels * 3, embed_dim=embedding_dim)
        #c1 32 c2 64
        self.localInf1 = localConv2d(c1_in_channels)
        self.localInf2 = localConv2d(c2_in_channels)
        self.localInf3 = localConv2d(c3_in_channels)
        self.cross = localADeepMerge(embedding_dim,2)
        self.linear_fuse = ConvModule(
            c1=embedding_dim*7,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        #c1 = 1,*32*64*64 经过localInf1变成1，32*3，32，32
        localI1 = self.localInf1(c1)
        localI2 = self.localInf2(c2)#c2 1,64,32,32  经过localInf2 1，64*3，16，16
        localI3 = self.localInf3(c3)

        _localI1 = self.linear_c1(localI1).permute(0,2,1).reshape(n, -1, localI1.shape[2], localI1.shape[3])
        _localI2 = self.linear_c2(localI2).permute(0,2,1).reshape(n, -1, localI2.shape[2], localI2.shape[3])
        _localI3 = self.linear_c3(localI3).permute(0,2,1).reshape(n, -1, localI3.shape[2], localI3.shape[3])

        #将深层的特征与浅层热进行cross attention
        Fix1 = self.cross(_localI1,c4)  #1,1024,256
        Fix2 = self.cross(_localI2,c4) #1,256,256
        Fix3 = self.cross(_localI3,c4)

        #将融合的模块和c1模块concat
        Fix1 = rearrange(Fix1, 'b (h w) c->b c h w',h=localI1.size()[2])   #256 ,32,32
        Fix2 = rearrange(Fix2, 'b (h w) c->b c h w',h=localI2.size()[2])#256,16,16
        Fix3 = rearrange(Fix3, 'b (h w) c->b c h w',h=localI3.size()[2])#256,8,8

        #将local和global块concat   16*16
        FixLG8=torch.cat([Fix3,_localI3],dim=1)
        FixLG16=torch.cat([Fix2,_localI2],dim=1)
        FixLG32=torch.cat([Fix1,_localI1],dim=1)

        FixLG8 = F.interpolate(FixLG8,size=c1.size()[2:],mode='bilinear', align_corners=False)
        FixLG16 = F.interpolate(FixLG16,size=c1.size()[2:],mode='bilinear', align_corners=False)
        FixLG32 = F.interpolate(FixLG32, size=c1.size()[2:], mode='bilinear', align_corners=False)
        #将16*16上采样到32*32
        GlobalM=F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        FixLGeve = torch.cat([FixLG8,FixLG16,FixLG32,GlobalM],dim=1)
        #将32*32的两个部分一起concat     上采样到64倍   #经过一个linear变成256

        FixLGeve =self.linear_fuse(FixLGeve)
        #dropout一下
        x = self.dropout(FixLGeve)
        # 分类头 变成1*numclass*原图像的1/4
        x = self.linear_pred(x)

        return x

class Fursformer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b2', pretrained = False):
        super(Fursformer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = FursformerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        #上采样恢复到原图大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

