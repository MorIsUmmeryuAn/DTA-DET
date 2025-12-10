import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels  # 默认输出通道与输入相同
        self.kernel_size = kernel_size
        self.num_heads = num_heads

        # 偏移量预测网络
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size * num_heads,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # 注意力权重预测网络
        self.attn_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * num_heads,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # 输出卷积
        self.output_conv = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.attn_conv.weight, 0.)
        nn.init.constant_(self.attn_conv.bias, 0.)

    def forward(self, x):
        B, C, H, W = x.shape

        # 预测偏移量
        offset = self.offset_conv(x)

        # 预测注意力权重
        attn = self.attn_conv(x)
        attn = attn.reshape(B, self.num_heads, self.kernel_size * self.kernel_size, H, W)
        attn = F.softmax(attn, dim=2)
        attn = attn.reshape(B, -1, H, W)

        # 应用可变形卷积
        out = deform_conv2d(
            x,
            offset,
            self.output_conv.weight,
            bias=self.output_conv.bias,
            padding=self.kernel_size // 2,
            mask=attn
        )

        return out