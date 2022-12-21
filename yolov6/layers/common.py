#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

from yolov6.layers.damo_yolo import ConvBNAct,RepGFPN,Focus,SuperResStem
from yolov6.layers.tiny_nas_csp import TinyNAS_CSP, TinyNAS_CSP_2
from yolov6.layers.CoAtNet import CoAtNetMBConv,ConvGE,CoAtNetTrans, MBConv_block, CoAtTrans_block
from yolov6.layers.focal_transformer import FocalTransformer_block
from yolov6.layers.BotNet import BotNet
from models.Models.research import BoT3
from models.common import C3,Add_down
from yolov6.layers.RTMDet import CSPNeXtLayer, RTM_SepBNHead, ConvModule,DepthwiseSeparableConv
from yolov6.layers.TAP import Task_aligned_Head
from yolov6.layers.yolov7 import E_ELAN,ELAN_H,ELAN, SPPCSPC
class SiLU(nn.Module):
    '''Activation of SiLU'''
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ConvWrapper(nn.Module):
    '''Wrapper for normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)

    def forward(self, x):
        return self.block(x)


class SimConvWrapper(nn.Module):
    '''Wrapper for normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = SimConv(in_channels, out_channels, kernel_size, stride, groups, bias)

    def forward(self, x):
        return self.block(x)


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher'''
    def __init__(self, in_channels, out_channels, kernel_size=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RealVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False,
    ):
        super(RealVGGBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.relu(self.se(self.bn(self.conv(inputs))))
        return out


class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        super(ScaleLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        init.constant_(self.weight, scale_init)
        self.num_features = num_features
        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)


#   A CSLA block is a LinearAddBlock with is_csla=True
class LinearAddBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False, is_csla=False, conv_scale_init=1.0):
        super(LinearAddBlock, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.scale_conv = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.scale_1x1 = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        if in_channels == out_channels and stride == 1:
            self.scale_identity = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_csla:     # Make them constant
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs)
        out = self.relu(self.se(self.bn(out)))
        return out


class DetectBackend(nn.Module):
    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):

        super().__init__()
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} format is not supported.'
        from yolov6.utils.checkpoint import load_checkpoint
        model = load_checkpoint(weights, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y, _ = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None
        elif block == BotRep:
            self.conv1 = BotRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BotRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_C3(nn.Module):
    '''Standard convolution in BepC3-Block'''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class BepC3(nn.Module):
    '''Beer-mug RepC3 Block'''
    def __init__(self, in_channels, out_channels, n=1,block=RepVGGBlock, e=0.5, concat=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))



class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_C3(c1, c_, 1, 1)
        self.cv2 = Conv_C3(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# class E_ELAN(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, n=1, k=2):
#         super(E_ELAN, self).__init__()
#         c_ = c1//2 # hidden channels
#
#         self.conv1 = Conv(c1,c_,1,1)
#         self.conv2 = Conv(c1,c_,1,1)
#         self.conv3 = Conv(c_, c_, 3, 1)
#         self.conv4 = Conv(c_, c_, 3, 1)
#         self.conv5 = Conv(c_, c_, 3, 1)
#         self.conv6 = Conv(c_, c_, 3, 1)
#         self.conv7 = Conv(c_, c_, 3, 1)
#         self.conv8 = Conv(c_, c_, 3, 1)
#         self.conv9 = Conv(c_ * 5, c2, 1, 1)
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)
#         x = torch.cat([x8, x6, x4, x2, x1], dim = 1)
#         x = self.conv9(x)
#         return x
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Out(nn.Module):
    def __init__(self, k=2):
        super(Out, self).__init__()

    def forward(self, x):
        outputs = []
        for y in x:
            outputs.append(y)
        return outputs

class MP1(nn.Module):
    def __init__(self, c2, idx = 1):  #c1 = c2
        c_ = c2//2
        self.idx = idx
        super(MP1, self).__init__()
        self.mp = MP()
        if idx == 1:
            self.conv1 = Conv(c2, c_, 1, 1)
            self.conv2 = Conv(c2, c_, 1, 1)
            self.conv3 = Conv(c_, c_, 3, 2)
        elif idx == 2:
            self.conv1 = Conv(c2, c2, 1, 1)
            self.conv2 = Conv(c2, c2, 1, 1)
            self.conv3 = Conv(c2, c2, 3, 2)

    def forward(self, input):
        x1 = self.mp(input)
        x1 = self.conv1(x1)
        x2 = self.conv2(input)
        x2 = self.conv3(x2)
        if self.idx == 1:
            out = torch.cat([x1, x2], dim=1)

        return out
class Stem(nn.Module):
    def __init__(self,in_channels,out_channels,block=RepVGGBlock):
        super(Stem, self).__init__()
        self.stem = block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )
    def forward(self,x):
        x = self.stem(x)
        return x
import math
class Head_layers(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_layers, self).__init__()

        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
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
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output

class Head_Depth(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_Depth, self).__init__()

        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # reg_conv0
        self.reg_conv = DepthwiseSeparableConv(in_chs=out_channels, out_chs=out_channels, dw_kernel_size=5, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
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
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)

        return x, cls_output, reg_output


class Head_simota(nn.Module):
    def __init__(self,in_channels,out_channels, reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_simota, self).__init__()
        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)

        self.obj_pred = nn.Conv2d(in_channels=out_channels, out_channels=1 * (num_anchors), kernel_size=1)
        self.prior_prob = 1e-2
        self.initialize_biases()
    def initialize_biases(self):
        self.na = 1
        b = self.cls_pred.bias.view(self.na, -1)
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.obj_pred.bias.view(self.na, -1)
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self,x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_pred(cls_feat)
        # cls_output = torch.sigmoid(cls_output)
        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(reg_feat)

        return cls_output, reg_output, obj_pred
class Head_out(nn.Module):
    def __init__(self,in_channels,out_channels,reg_max = 16,num_classes = 3, num_anchors = 1):
        super(Head_out, self).__init__()
        self.stem = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        # cls_conv0
        self.cls_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # reg_conv0
        self.reg_conv = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        # cls_pred0
        self.cls_pred = nn.Conv2d(in_channels=out_channels, out_channels=num_classes * num_anchors, kernel_size=1)
        # reg_pred0
        self.reg_pred = nn.Conv2d(in_channels=out_channels, out_channels=4 * (reg_max + num_anchors), kernel_size=1)
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
        cls_x = x
        reg_x = x
        cls_output = self.cls_pred(cls_x)
        cls_output = torch.sigmoid(cls_output)
        reg_output = self.reg_pred(reg_x)

        return x, cls_output, reg_output

# BoT
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        # print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                        content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            # print('new pos222-> shape:',content_position.shape)
            # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    # expansion = 1

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        self.fc1 = nn.Linear(c2, c2)

    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out


# class BoT3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):  # ch_in, ch_out, number, , expansion,w,h
#         super(BoT3, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(
#             *[BottleneckTransformer(c_, c_, stride=1, heads=4, mhsa=True, resolution=(w, h), expansion=e2) for _ in
#               range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

from timm.models.layers import DropPath
class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
'''RepVGGBlock is a basic rep-style block, including training and deploy status
This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
'''
class RepGhostVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepGhostVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


#############RepGhost
# RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization
# https://github.com/ChengpengChen/RepGhost/blob/main/model/repghost.py
class RepGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True
class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        dw_kernel_size=3,
        stride=1,
        c = 16,
        exp_size = 8,
        width = 0.5,
        se_ratio=0.0,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        out_chs = _make_divisible(c * width, 4)
        mid_chs = _make_divisible(exp_size * width, 4)
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
import copy
def repghost_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
class RepGhostNet(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        super(RepGhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.num_classes = num_classes

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = RepGhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        shortcut=shortcut,
                        reparam=reparam,
                        reparam_bn=reparam_bn,
                        reparam_identity=reparam_identity,
                        deploy=deploy
                    ),
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width * 2, 4)
        stages.append(
            nn.Sequential(
                ConvBnAct(input_channel, output_channel, 1),
            ),
        )
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(
            input_channel, output_channel, 1, 1, 0, bias=True,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)

#build backbone
def repghostnet(enable_se=True, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ]

    return RepGhostNet(cfgs, **kwargs)


############### end of repghost
#######focal#####
#focal_levels = [2, 2, 2, 2]  focal_windows=7, #[7, 5, 3, 1] depths [2,2,6,2]
class FocalTransformer(nn.Module):
    def __init__(self,in_chans, out_chans,depths=2,focal_windows=7,window_size=7):
        super(FocalTransformer, self).__init__()

        num_heads = out_chans // 32
        self.focal_transformer = FocalTransformer_block(in_chans = in_chans, embed_dim = out_chans,depths = depths,
                                                        num_heads = num_heads,focal_windows = focal_windows,window_size = window_size)
    def forward(self,x):
        x = self.focal_transformer(x)
        return x
class FocalC3(BepC3):
    def __init__(self, in_channels, out_channels, depths=2, focal_windows=7,window_size=7,e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * e)
        num_heads = out_channels // 32
        self.m = FocalTransformer_block(in_chans = c_, embed_dim = c_,depths = depths,
                                                        num_heads = num_heads,focal_windows = focal_windows,window_size = window_size)
class MBConvC3(BepC3):
    def __init__(self, in_channels, out_channels, num_blocks=2, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * e)
        self.m = MBConv_block(in_channels = c_, out_channels = c_,num_blocks = num_blocks)
class RepGhostC3(BepC3):
    def __init__(self, in_channels, out_channels,mid_c, num_blocks = 1, dw_kernel_size = 3,se_ratio = 0.0,width=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(in_channels, out_channels)
        c_ = int(out_channels * width)
        self.m = nn.Sequential(*(RepGhostBottleneck(in_chs = c_,exp_size = mid_c, c = int(c_/width),width = width,se_ratio = se_ratio,dw_kernel_size = dw_kernel_size) for _ in range(num_blocks)))

class BepBotC3(nn.Module):
    '''Beer-mug RepC3 Block'''

    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, e=0.5,
                 concat=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BotRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))
from yolov6.layers.BotNet import Attention
class BotRep(nn.Module):

    def __init__(self, in_channels, out_channels, head = 4,dim_head = 128,rel_pos_emb = False, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        attn_dim_out = head * dim_head
        # self.conv1 = basic_block(in_channels, out_channels)
        self.botAttention = nn.Sequential(
            Attention(
                dim = out_channels,
                heads = head,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            nn.ReLU(),
        )
        self.conv2 = basic_block(attn_dim_out, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        # outputs = self.conv1(x)
        outputs = self.botAttention(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return SimConvWrapper
    elif mode == 'conv_silu':
        return ConvWrapper
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))
