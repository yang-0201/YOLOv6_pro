import torch
import torch.nn as nn
from timm.models.layers import create_conv2d, DropPath, get_norm_act_layer
import numpy as np
def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class ConvModule(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=True, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out

class CSPNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels, out_channels, expansion=0.5, add_identity=True,use_depthwise = True):
        super(CSPNeXtBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_depth = DepthwiseSeparableConv

        self.conv1 = ConvModule(in_channels, hidden_channels, k = 3, s=1)
        if use_depthwise:
            self.conv2 = conv_depth(hidden_channels, out_channels, 5, stride=1)
        else:
            self.conv2 = ConvModule(hidden_channels, out_channels, 5, s=1)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPNeXtLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 channel_attention = True,
                 add_identity=True,
                 use_depthwise=True,
                 expand_ratio=0.5):
        super(CSPNeXtLayer, self).__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)

        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)
        self.channel_attention = channel_attention
        if self.channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity, use_depthwise) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)
import math
# RTMDetHead with separated BN layers and shared conv layers.
class RTM_SepBNHead(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, stage = 3,stacked_convs_number = 2, num_anchors = 1,share_conv = True):
        super(RTM_SepBNHead, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.stage = stage
        self.stacked_convs_number = stacked_convs_number

        for n in range(self.stage):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs_number):
                chn = in_channels[n] if i == 0 else out_channels[i]
                cls_convs.append(
                    ConvModule(
                        chn,
                        out_channels[n],
                        3,
                        s=1,
                        p=1 ))
                reg_convs.append(
                    ConvModule(
                        chn,
                        out_channels[n],
                        3,
                        s=1,
                        p=1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.rtm_cls.append(
                nn.Conv2d(
                    out_channels[n],
                    num_classes * num_anchors,
                    1,
                    padding=0))
            self.rtm_reg.append(
                nn.Conv2d(
                    out_channels[n],
                    4 * (reg_max + num_anchors),
                    1,
                    padding=0))
        if share_conv:
            for n in range(stage):
                for i in range(self.stacked_convs_number):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self.initialize_biases()


    def  initialize_biases(self):


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if isinstance(m,ConvModule):
                constant_init(m.bn, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)

    def forward(self,feats):
        cls_scores = []
        bbox_preds = []
        outputs = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            #reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            reg_dist = self.rtm_reg[idx](reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            cls_score = torch.sigmoid(cls_score)
            output = [feats[idx],cls_score,reg_dist]
            outputs.append(output)

        return outputs

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init