# dda_wa.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDualAttentionWA(nn.Module):
    """Dynamic Dual-Attention Weight Adaptation Module"""

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(channels // reduction_ratio, 4)

        # Channel Attention Path
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features=channels, out_features=self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.reduced_channels, out_features=channels),
            nn.Sigmoid()
        )

        # 确保线性层的输入输出维度正确
        print(f"DDAWA initialized with channels: {channels}, reduced: {self.reduced_channels}")

        # Spatial Attention Path
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        # 验证输入通道数与预期一致
        if c != self.channels:
            raise ValueError(f"Input channels {c} don't match initialized channels {self.channels}")

        # Channel branch
        gap = F.adaptive_avg_pool2d(x, 1)
        gap = gap.view(b, c)
        channel_weights = self.channel_attention(gap).view(b, c, 1, 1)

        # Spatial branch
        spatial_weights = self.spatial_attention(x)

        # Fusion
        fused_weights = torch.sigmoid(
            self.alpha * (channel_weights * spatial_weights) + self.beta
        )

        return x * fused_weights


class ConcatWithDDAWA(nn.Module):
    """Concat that applies DDAWA to inputs first"""

    def __init__(self, dim=1, *args):
        super().__init__()
        self.dim = dim
        self.preset_channels = list(args) if args else []
        self.ddawa_modules = nn.ModuleList()
        self.initialized = False
        print(f"ConcatWithDDAWA initialized with dim={self.dim}, preset_channels={self.preset_channels}")

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            return x

        # 第一次运行时，根据输入特征或预设通道数创建DDAWA模块
        if not self.initialized:
            if len(self.preset_channels) > 0:
                # 检查预设通道数与输入特征是否匹配
                if len(self.preset_channels) != len(x):
                    raise ValueError(
                        f"Preset channels {self.preset_channels} don't match input features {[f.shape for f in x]}")

                print(f"Initializing DDAWA modules with preset channels: {self.preset_channels}")
                for i, (feature, preset_c) in enumerate(zip(x, self.preset_channels)):
                    actual_c = feature.size(1)
                    if preset_c != actual_c:
                        print(
                            f"Warning: Overriding preset channel {preset_c} with actual channel {actual_c} at index {i}")
                    self.ddawa_modules.append(DynamicDualAttentionWA(channels=actual_c))
            else:
                # 根据输入特征动态初始化
                print(f"Initializing DDAWA modules for inputs with shapes: {[feat.shape for feat in x]}")
                for feature in x:
                    channels = feature.size(1)
                    self.ddawa_modules.append(DynamicDualAttentionWA(channels=channels))
            self.initialized = True

        weighted_inputs = []
        for i, feature in enumerate(x):
            weighted_feature = self.ddawa_modules[i](feature)
            weighted_inputs.append(weighted_feature)

        return torch.cat(weighted_inputs, dim=self.dim)