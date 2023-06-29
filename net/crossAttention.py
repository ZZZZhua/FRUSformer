
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, reduce,repeat

class FeedForward(nn.Module):
    """
    Feedforward层
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim, num_heads=2):
        super(MultiHeadCrossAttention, self).__init__()
        self.in_dim = in_dim  # 输入特征的深度
        self.num_heads = num_heads  # 注意力头的数量

        # 将输入特征深度in_dim分为num_heads份，给每个注意力头使用
        assert in_dim % num_heads == 0
        self.head_dim = int(in_dim / num_heads)

        # 三个子层的线性变换
        self.query_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.key_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.value_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.final_linear = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self,  tgt,src):
        batch_size = src.size(0)
        src = rearrange(src, 'b c h w -> b (h w) c')
        tgt = rearrange(tgt, 'b c h w -> b (h w) c')

        query = self.query_linear(tgt).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(src).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(src).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: [batch_size, num_heads, tgt_len, src_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        # 注意力权重: [batch_size, num_heads, tgt_len, src_len]
        attention_weights = F.softmax(attention_scores, dim=-1)


        weighted_src = torch.matmul(attention_weights, value)


        weighted_src = weighted_src.transpose(1, 2).contiguous().view(batch_size, -1, self.in_dim)

        output = self.final_linear(weighted_src + tgt)
        return output


class localADeepMerge(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """

        Args:
            d_model:
            num_heads:
            dropout:
        """
        super().__init__()
        self.attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,y):
        """
        x的形状为(batch_size, seq_len, d_model)
        """
        # self-attention部分
        att_output = self.attention(x, y)
        att_output = self.dropout1(att_output)
        att_output = self.norm1(att_output + att_output)

        # FFN部分
        ffn_output = self.ff(att_output)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.norm2(ffn_output + att_output)

        return ffn_output

