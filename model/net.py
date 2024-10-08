"""
实现IFF方法
可调节攻击幅度
"""

import torch
import torch.nn as nn
from typing import Union, List
from PIL import Image
import numpy as np
#from   thop import profile, clever_format
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.norm(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = self.norm(out)
        return out


class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca_layer = ChannelAttention(out_channels)
        self.sa_layer = SpatialAttention()
        self.ca_assigned = None
        self.sa_assigned = None
        self.ca_reserved = None
        self.sa_reserved = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.ca_assigned is None:
            ca = self.ca_layer(out)
        else:
            ca = self.ca_assigned
        if self.sa_assigned is None:
            sa = self.sa_layer(out)
        else:
            sa = self.sa_assigned
        self.ca_reserved = ca
        self.sa_reserved = sa
        out = ca * out
        out = sa * out
        out += residual
        out = self.relu(out)
        return out
    

class Res_SP_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_SP_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.sa_layer = SpatialAttention()
        self.sa_assigned = None
        self.sa_reserved = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.sa_assigned is None:
            sa = self.sa_layer(out)
        else:
            sa = self.sa_assigned
        self.sa_reserved = sa
        out = sa * out
        out += residual
        out = self.relu(out)
        return out


class LightWeightNetwork(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128]):
        super(LightWeightNetwork, self).__init__()
        if block == 'Res_block':
            block = Res_block
        elif block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class LightWeightNetwork_IFF(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], iff_back_num=1):
        super(LightWeightNetwork_IFF, self).__init__()
        if block == 'Res_block':
            block = Res_block
        elif block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.iff_conv = nn.Conv2d(num_classes, nb_filter[0], kernel_size=3, padding=1)
        self.iff_back_num = iff_back_num

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        for i in range(self.iff_back_num):
            output_back = self.iff_conv(nn.functional.relu(output))
            output = self.final(x0_4+output_back)

        return output


class LightWeightNetwork_AAL_1(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_1, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = back_mask
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_2(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_2, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = one-back_mask
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_3(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_3, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = self.sa_norm(back_mask + sa)
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(backtracked_sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_4(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_4, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = self.sa_norm((0.2*one + 0.8*back_mask) + sa)
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(backtracked_sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None

    
class LightWeightNetwork_AAL_5(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_5, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = one - self.sa_norm((0.2*one + 0.8*back_mask) + sa)
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(backtracked_sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output    

    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_6(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_6, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = one - self.sa_norm(back_mask + sa)
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(backtracked_sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_7(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_7, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = one-back_mask
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_AAL_8(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_8, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
        self.reserved_delta = adv_delta

        sa = self.get_sa_reserved(self.conv0_4).detach()
        self.reserved_sa = sa

        ### backtracking ###
        delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
        back_mask = mask_top_rate(delta_channel_max, 0.01)
        back_mask = self.mask_pool(back_mask)
        self.reserved_back_mask = back_mask
        one = torch.ones_like(back_mask)
        backtracked_sa = one-back_mask
        backtracked_sa = backtracked_sa.detach()
        self.reserved_backtracked_sa = backtracked_sa
        ### end backtracking ###

        self.reserved_attacked_img = input+self.eps*adv_delta
        self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
        self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

        self.assign_sa(backtracked_sa, self.conv0_0)
        x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
        self.reset_sa(self.conv0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None



class LightWeightNetwork_AAL_9(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_AAL_9, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sa_norm = nn.BatchNorm2d(1)
        self.mask_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps
        self.train_iter = 0

        # reserved(imgs, attentions)
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None
        self.reserved_attacked_img = None
        self.reserved_attacked_img_sa = None
        self.reserved_attacked_img_backtracked_sa = None
        self.reserved_back_mask = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        if self.train_iter % 2 == 0:
            self.train_iter += 1
            adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
            x0_0 = self.conv0_0(input+adv_delta)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))
            x3_0 = self.conv3_0(self.pool(x2_0))
            x4_0 = self.conv4_0(self.pool(x3_0))

            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

            output = self.final(x0_4)
            loss = criterion(output, target)
            loss:torch.Tensor
            loss.backward()

            adv_delta_grad = adv_delta.grad
            adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()
            self.reserved_delta = adv_delta

            sa = self.get_sa_reserved(self.conv0_4).detach()
            self.reserved_sa = sa

            ### backtracking ###
            delta_channel_max, _ = torch.max(torch.abs(adv_delta_grad), dim=1, keepdim=True)
            back_mask = mask_top_rate(delta_channel_max, 0.01)
            back_mask = self.mask_pool(back_mask)
            self.reserved_back_mask = back_mask
            one = torch.ones_like(back_mask)
            backtracked_sa = one-back_mask
            backtracked_sa = backtracked_sa.detach()
            self.reserved_backtracked_sa = backtracked_sa
            ### end backtracking ###

            self.reserved_attacked_img = input+self.eps*adv_delta
            self.reserved_attacked_img_sa = input+self.eps*sa*adv_delta
            self.reserved_attacked_img_backtracked_sa = input+self.eps*backtracked_sa*adv_delta

            x0_0 = self.conv0_0(input+self.eps*backtracked_sa*adv_delta)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))
            x3_0 = self.conv3_0(self.pool(x2_0))
            x4_0 = self.conv4_0(self.pool(x3_0))

            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            self.assign_sa(sa, self.conv0_4)
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            self.reset_sa(self.conv0_4)

            output = self.final(x0_4)
        else:
            self.train_iter += 1
            output = self.forward_test(input)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


def mask_top_rate(data, kept_rate):
    """
    Given a batch of samples(in a tensor form) and a rate of kept values,
    return a mask tensor which has the same shape as the input tensor. And
    the positions where the input tensor's value is within its topk range
    determined by the kept rate are set to be 1, others are set to be 0.

    Args:
        data (Tensor): The input samples
        kept_rate (float): The kept rate. Of each sample in the batch, how
            much ratio of the values from top are kept.

    Returns:
        Tensor: The mask tensor in the same shape as the input tensor.
    """
    data_flattened = data.view(data.size(0), -1)
    values_kept, _ = data_flattened.topk(int(data_flattened.size(1)*kept_rate), dim=1)
    values_min, _ = torch.min(values_kept, dim=-1)
    values_min = values_min.unsqueeze(-1).repeat(1, data_flattened.size(-1))
    mask = torch.ge(data_flattened, values_min).float().view(data.size())
    return mask


class LightWeightNetwork_FGSM(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_FGSM, self).__init__()
        if block == 'Res_block':
            block = Res_block
        elif block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()

        x0_0 = self.conv0_0(input+self.eps*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)


class LightWeightNetwork_FGSM_SA(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_FGSM_SA, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor, target, criterion):
        adv_delta = torch.zeros_like(input).to(input.device).requires_grad_(True)
        x0_0 = self.conv0_0(input+adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_delta_grad = adv_delta.grad
        adv_delta = torch.sign(adv_delta_grad).to(input.device).detach()

        sa = self.get_sa_reserved(self.conv0_4).detach()

        x0_0 = self.conv0_0(input+self.eps*sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input, target=None, criterion=None):
        if self.training:
            return self.forward_train(input, target, criterion)
        else:
            return self.forward_test(input)

    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_SA(nn.Module):
    """SA: Selective Attacks"""
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_SA, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        adv_delta = torch.randn(input.size()).to(input.device)
        sa = self.get_sa_reserved(self.conv0_4)

        x0_0 = self.conv0_0(input+self.eps*sa*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        self.assign_sa(sa, self.conv0_4)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        self.reset_sa(self.conv0_4)

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_SA_2(nn.Module):
    """SA: Selective Attacks"""
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_SA_2, self).__init__()
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        adv_delta_3 = torch.randn(self.pool(x2_0).size()).to(input.device)
        adv_delta_4 = torch.randn(self.pool(x3_0).size()).to(input.device)
        sa_3 = self.get_sa_reserved(self.conv3_0)
        sa_4 = self.get_sa_reserved(self.conv4_0)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0)+sa_3*self.eps*adv_delta_3)
        x4_0 = self.conv4_0(self.pool(x3_0)+sa_4*self.eps*adv_delta_4)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_test(input)
    
    def get_sa_reserved(self, conv_layer: nn.Sequential) -> torch.Tensor:
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        sa_reserved = res_block.sa_reserved
        return sa_reserved
    
    def assign_sa(self, sa, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = sa

    def reset_sa(self, conv_layer: nn.Sequential):
        res_block = list(conv_layer.children())[-1]
        res_block: Union[Res_CBAM_block, Res_SP_block]
        res_block.sa_assigned = None


class LightWeightNetwork_RN(nn.Module):
    """SA: Selective Attacks"""
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], eps=0.01):
        super(LightWeightNetwork_RN, self).__init__()
        if block == 'Res_block':
            block = Res_block
        elif block == 'Res_CBAM_block':
            block = Res_CBAM_block
        elif block == 'Res_SP_block':
            block = Res_SP_block
        else:
            raise ValueError(f"Block type {block} is not supported in {type(self)}")

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.eps = eps

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input:torch.Tensor):
        adv_delta = torch.randn(input.size()).to(input.device)
        x0_0 = self.conv0_0(input+self.eps*adv_delta)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output
    
    def forward_test(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_test(input)

# # #####################################
# # ### FLops, Params, Inference time evaluation
# if __name__ == '__main__':
#     from model.load_param_data import  load_param
#     import time
#     import os
#     from torchstat import stat
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#     nb_filter, num_blocks= load_param('two', 'resnet_18')
#     input       = torch.randn(1, 3, 256, 256,).cuda()
#     in_channels = 3
#     # model   = res_UNet(num_classes=1, input_channels=in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter)
#     model       = LightWeightNetwork(num_classes=1, input_channels=in_channels, block=Res_block, num_blocks=num_blocks, nb_filter=nb_filter)
#     a           = stat(model, (3,256,256))
#     # model = model.cuda()
#     # flops, params = profile(model, inputs=(input,), verbose=True)
#     # flops, params = clever_format([flops, params], "%.3f")
#     # start_time = time.time()
#     # output     = model(input)
#     # end_time   = time.time()
#     # print('flops:', flops, 'params:', params)
#     # print('inference time per image:',end_time-start_time )
