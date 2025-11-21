import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['M2FormerBlocks']


# 1. Conv + BN + Act
class ConvBNAct(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0,
                 groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 2. Pointwise Conv
class PWConv(ConvBNAct):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)


# 3. Depthwise Conv
class DWConv(ConvBNAct):
    def __init__(self, channels, stride=1):
        super().__init__(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels)


# 4. DSConv（Depthwise Separable Conv)
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.dsconv = nn.Sequential(
            DWConv(in_channels, stride=stride),
            PWConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.dsconv(x)


# 5. SPDConv（downsampling）
class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = PWConv(in_channels * 4, out_channels)

    def forward(self, x):
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.conv(x)


# 6. CBAM Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * attn


# 7. Efficient Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x).view(x.size(0), 1, x.size(1))
        attn = self.sigmoid(self.conv(out)).view(x.size(0), x.size(1), 1, 1)
        return x * attn


# 8. Channel Mixer（Conv-mlp + ECA）
class ChannelMixer(nn.Module):
    def __init__(self, channels, expansion=2, att=True):
        super().__init__()
        mid = channels * expansion
        self.conv1 = PWConv(channels, mid)
        self.conv2 = PWConv(mid, channels)
        self.att = att
        self.channel_attn = ChannelAttention(channels) if att else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        return identity + self.channel_attn(out)


# 9. Token Mixer（Res2Net + SA）
class TokenMixer(nn.Module):
    def __init__(self, channels, scale=4, att=True):
        super().__init__()
        assert channels % scale == 0
        self.depth = channels // scale
        self.scale = scale
        self.convs = nn.ModuleList([
            ConvBNAct(self.depth, self.depth, kernel_size=3, padding=1)
            for _ in range(scale - 1)
        ])
        self.norm_act = nn.Sequential(nn.BatchNorm2d(channels), nn.SiLU())
        self.att = att
        self.spatial_attn = SpatialAttention() if att else nn.Identity()

    def forward(self, x):
        identity = x
        splits = torch.chunk(x, self.scale, dim=1)
        out = [splits[0]]
        for i in range(1, self.scale):
            out.append(self.convs[i - 1](splits[i] + out[-1]))
        out = torch.cat(out, dim=1)
        out = self.norm_act(out)
        return identity + self.spatial_attn(out)

# # 9. Token Mixer（common conv + SA）
# class TokenMixer(nn.Module):
#     def __init__(self, channels, scale=4, att=True):
#         super().__init__()
#         self.scale = scale
#         self.conv = ConvBNAct(channels, channels, kernel_size=3, padding=1)
#         self.norm_act = nn.Sequential(nn.BatchNorm2d(channels), nn.SiLU())
#         self.att = att
#         self.spatial_attn = SpatialAttention() if att else nn.Identity()
#
#     def forward(self, x):
#         identity = x
#         out = self.conv(x)
#         out = self.norm_act(out)
#         return identity + self.spatial_attn(out)


# 10. M2FormerBlock
class M2FormerBlock(nn.Module):
    def __init__(self, channels, scale=4, sp_att=True, ch_att=True):
        super().__init__()
        self.token_mixer = TokenMixer(channels, scale, att=sp_att)
        self.channel_mixer = ChannelMixer(channels, att=ch_att)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


# 11. BasicBlock（M2Former, DSConv, Shortcut）
class M2FormerBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, scale=4, sp_att=True, ch_att=True, use_spd=True):
        super().__init__()
        self.m2former = M2FormerBlock(in_ch, scale=scale, sp_att=sp_att, ch_att=ch_att)
        self.downsample_main = DSConv(in_ch, out_ch, stride=stride) if stride == 2 else PWConv(in_ch, out_ch)
        self.shortcut = SPDConv(in_ch, out_ch) if (stride == 2 and use_spd) else (
            ConvBNAct(in_ch, out_ch, kernel_size=1, stride=2, padding=0) if stride == 2 else nn.Identity())

    def forward(self, x):
        out = self.m2former(x)
        out = self.downsample_main(out)
        out = out + self.shortcut(x)
        return F.silu(out)


# 12. Blocks wrapper
class M2FormerBlocks(nn.Module):
    def __init__(self, ch_in, ch_out, count, block, stage_num, sp_att=True, ch_att=True, use_spd=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        block = globals()[block]
        for i in range(count):
            self.blocks.append(
                block(
                    in_ch=ch_in,
                    out_ch=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    scale=4,
                    sp_att=sp_att,
                    ch_att=ch_att,
                    use_spd=use_spd
                )
            )
            ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
