import torch
import torch.nn as nn
from timm.models.layers import create_conv2d, DropPath, get_norm_act_layer

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
class Head_RTM(nn.Module):
    def __init__(self,in_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_RTM, self).__init__()

        # self.stem = ConvModule(c1=in_channels, c2=out_channels, k=3, s=1)
        # cls_conv
        self.cls_conv1 = ConvModule(c1=in_channels, c2=in_channels, k=3, s=1)
        self.cls_conv2 = ConvModule(c1=in_channels, c2=in_channels, k=3, s=1)
        # reg_conv
        self.reg_conv1 = ConvModule(c1=in_channels, c2=in_channels, k=3, s=1)
        self.reg_conv2 = ConvModule(c1=in_channels, c2=in_channels, k=3, s=1)
        # cls_pred
        self.cls_pred = nn.Conv2d(in_channels=in_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=in_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()

    def initialize_biases(self):


        b = self.cls_pred.bias.view(-1, )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.cls_pred.weight
        w.data.fill_(0.)
        self.cls_pred.weight = torch.nn.Parameter(w, requires_grad=True)


        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self,x):
        # x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv1(cls_x)
        cls_feat = self.cls_conv2(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        cls_output = torch.sigmoid(cls_output)  ######

        reg_feat = self.reg_conv1(reg_x)
        reg_feat = self.reg_conv2(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output