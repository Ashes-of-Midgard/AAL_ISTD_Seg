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


class LightWeightNetwork_AAL(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], attack_layer_ids=[0]):
        super(LightWeightNetwork_AAL, self).__init__()
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

        self.attack_layer_ids = attack_layer_ids

        # reserved(imgs, attentions)
        self.reserved_img = None
        self.reserved_sa = None
        self.reserved_backtracked_sa = None
        self.reserved_delta = None

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input, target, criterion):
        self.reserved_img = input
        init_deltas = [torch.zeros_like(input).requires_grad_()]
        x0_0 = self.conv0_0(input+init_deltas[0])
        sa0_0 = self.get_sa_reserved(self.conv0_0)

        x0_0_pooled = self.pool(x0_0)
        init_deltas.append(torch.zeros_like(x0_0_pooled).requires_grad_())
        x1_0 = self.conv1_0(x0_0_pooled+init_deltas[1])
        sa1_0 = self.get_sa_reserved(self.conv1_0)

        x1_0_pooled = self.pool(x1_0)
        init_deltas.append(torch.zeros_like(x1_0_pooled).requires_grad_())
        x2_0 = self.conv2_0(x1_0_pooled+init_deltas[2])
        sa2_0 = self.get_sa_reserved(self.conv2_0)

        x2_0_pooled = self.pool(x2_0)
        init_deltas.append(torch.zeros_like(x2_0_pooled).requires_grad_())
        x3_0 = self.conv3_0(x2_0_pooled+init_deltas[3])
        sa3_0 = self.get_sa_reserved(self.conv3_0)

        x3_0_pooled = self.pool(x3_0)
        init_deltas.append(torch.zeros_like(x3_0_pooled).requires_grad_())
        x4_0 = self.conv4_0(x3_0_pooled+init_deltas[4])
        sa4_0 = self.get_sa_reserved(self.conv4_0)

        sa_list = [sa0_0, sa1_0, sa2_0, sa3_0, sa4_0]

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_deltas = [None, None, None, None, None]
        backtracked_sa = [None, None, None, None, None]
        for layer_id in self.attack_layer_ids:
            init_delta = init_deltas[layer_id]
            init_delta:torch.Tensor
            delta = 0.03 * torch.sign(init_delta.grad).detach()
            self.reserved_delta = torch.sign(init_delta.grad).detach()
            
            sa = sa_list[layer_id]
            sa:torch.Tensor
            self.reserved_sa = sa.detach()
            delta_channel_max, _ = torch.max(delta, dim=1, keepdim=True)
            back_mask = mask_top_rate(delta_channel_max, 0.01)
            one = torch.ones_like(back_mask)

            adv_deltas[layer_id] = delta
            backtracked_sa[layer_id] = (one - 0.05 * back_mask) * sa.detach()
            self.reserved_backtracked_sa = backtracked_sa[layer_id]

        if 0 in self.attack_layer_ids:
            input = input + backtracked_sa[0] * adv_deltas[0]
            self.assign_sa(backtracked_sa[0], self.conv0_0)
        x0_0 = self.conv0_0(input)
        self.reset_sa(self.conv0_0)
        
        x0_0_pooled = self.pool(x0_0)
        if 1 in self.attack_layer_ids:
            x0_0_pooled = x0_0_pooled + backtracked_sa[1] * adv_deltas[1]
            self.assign_sa(backtracked_sa[1], self.conv1_0)
        x1_0 = self.conv1_0(x0_0_pooled)
        self.reset_sa(self.conv1_0)
        
        x1_0_pooled = self.pool(x1_0)
        if 2 in self.attack_layer_ids:
            x1_0_pooled = x1_0_pooled + backtracked_sa[2] * adv_deltas[2]
            self.assign_sa(backtracked_sa[2], self.conv2_0)
        x2_0 = self.conv2_0(x1_0_pooled)
        self.reset_sa(self.conv2_0)

        x2_0_pooled = self.pool(x2_0)
        if 3 in self.attack_layer_ids:
            x2_0_pooled = x2_0_pooled + backtracked_sa[3] * adv_deltas[3]
            self.assign_sa(backtracked_sa[3], self.conv3_0)
        x3_0 = self.conv3_0(x2_0_pooled)
        self.reset_sa(self.conv3_0)

        x3_0_pooled = self.pool(x3_0)
        if 4 in self.attack_layer_ids:
            x3_0_pooled = x3_0_pooled + backtracked_sa[4] * adv_deltas[4]
            self.assign_sa(backtracked_sa[4], self.conv4_0)
        x4_0 = self.conv4_0(x3_0_pooled)
        self.reset_sa(self.conv4_0)

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
    
    def get_sa_reserved(self, conv_layer: nn.Sequential):
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
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], attack_layer_ids=[0]):
        super(LightWeightNetwork_FGSM, self).__init__()
        if block == 'Res_block':
            block = Res_block
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

        self.attack_layer_ids = attack_layer_ids

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input, target, criterion):
        init_deltas = [torch.zeros_like(input).requires_grad_()]
        x0_0 = self.conv0_0(input+init_deltas[0])

        x0_0_pooled = self.pool(x0_0)
        init_deltas.append(torch.zeros_like(x0_0_pooled).requires_grad_())
        x1_0 = self.conv1_0(x0_0_pooled+init_deltas[1])

        x1_0_pooled = self.pool(x1_0)
        init_deltas.append(torch.zeros_like(x1_0_pooled).requires_grad_())
        x2_0 = self.conv2_0(x1_0_pooled+init_deltas[2])

        x2_0_pooled = self.pool(x2_0)
        init_deltas.append(torch.zeros_like(x2_0_pooled).requires_grad_())
        x3_0 = self.conv3_0(x2_0_pooled+init_deltas[3])

        x3_0_pooled = self.pool(x3_0)
        init_deltas.append(torch.zeros_like(x3_0_pooled).requires_grad_())
        x4_0 = self.conv4_0(x3_0_pooled+init_deltas[4])

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_deltas = [None, None, None, None, None]
        for layer_id in self.attack_layer_ids:
            init_delta = init_deltas[layer_id]
            init_delta:torch.Tensor
            delta = 0.03 * torch.sign(init_delta.grad).detach()
            adv_deltas[layer_id] = delta

        if 0 in self.attack_layer_ids:
            input = input + adv_deltas[0]
        x0_0 = self.conv0_0(input)
        
        x0_0_pooled = self.pool(x0_0)
        if 1 in self.attack_layer_ids:
            x0_0_pooled = x0_0_pooled + adv_deltas[1]
        x1_0 = self.conv1_0(x0_0_pooled)
        
        x1_0_pooled = self.pool(x1_0)
        if 2 in self.attack_layer_ids:
            x1_0_pooled = x1_0_pooled + adv_deltas[2]
        x2_0 = self.conv2_0(x1_0_pooled)

        x2_0_pooled = self.pool(x2_0)
        if 3 in self.attack_layer_ids:
            x2_0_pooled = x2_0_pooled + adv_deltas[3]
        x3_0 = self.conv3_0(x2_0_pooled)

        x3_0_pooled = self.pool(x3_0)
        if 4 in self.attack_layer_ids:
            x3_0_pooled = x3_0_pooled + adv_deltas[4]
        x4_0 = self.conv4_0(x3_0_pooled)

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
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], attack_layer_ids=[3,4]):
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

        self.attack_layer_ids = attack_layer_ids

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input, target, criterion):
        init_deltas = [torch.zeros_like(input).requires_grad_()]
        x0_0 = self.conv0_0(input+init_deltas[0])
        sa0_0 = self.get_sa_reserved(self.conv0_0)

        x0_0_pooled = self.pool(x0_0)
        init_deltas.append(torch.zeros_like(x0_0_pooled).requires_grad_())
        x1_0 = self.conv1_0(x0_0_pooled+init_deltas[1])
        sa1_0 = self.get_sa_reserved(self.conv1_0)

        x1_0_pooled = self.pool(x1_0)
        init_deltas.append(torch.zeros_like(x1_0_pooled).requires_grad_())
        x2_0 = self.conv2_0(x1_0_pooled+init_deltas[2])
        sa2_0 = self.get_sa_reserved(self.conv2_0)

        x2_0_pooled = self.pool(x2_0)
        init_deltas.append(torch.zeros_like(x2_0_pooled).requires_grad_())
        x3_0 = self.conv3_0(x2_0_pooled+init_deltas[3])
        sa3_0 = self.get_sa_reserved(self.conv3_0)

        x3_0_pooled = self.pool(x3_0)
        init_deltas.append(torch.zeros_like(x3_0_pooled).requires_grad_())
        x4_0 = self.conv4_0(x3_0_pooled+init_deltas[4])
        sa4_0 = self.get_sa_reserved(self.conv4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        loss = criterion(output, target)
        loss:torch.Tensor
        loss.backward()

        adv_deltas = [None, None, None, None, None]
        for layer_id in self.attack_layer_ids:
            init_delta = init_deltas[layer_id]
            init_delta:torch.Tensor
            delta = 0.03 * torch.sign(init_delta.grad).detach()
            adv_deltas[layer_id] = delta

        if 0 in self.attack_layer_ids:
            input = input + sa0_0.detach() * adv_deltas[0]
            self.assign_sa(sa0_0, self.conv0_0)
        x0_0 = self.conv0_0(input)
        self.reset_sa(self.conv0_0)
        
        x0_0_pooled = self.pool(x0_0)
        if 1 in self.attack_layer_ids:
            x0_0_pooled = x0_0_pooled + sa1_0.detach() * adv_deltas[1]
            self.assign_sa(sa1_0, self.conv1_0)
        x1_0 = self.conv1_0(x0_0_pooled)
        self.reset_sa(self.conv1_0)
        
        x1_0_pooled = self.pool(x1_0)
        if 2 in self.attack_layer_ids:
            x1_0_pooled = x1_0_pooled + sa2_0.detach() * adv_deltas[2]
            self.assign_sa(sa2_0, self.conv2_0)
        x2_0 = self.conv2_0(x1_0_pooled)
        self.reset_sa(self.conv2_0)

        x2_0_pooled = self.pool(x2_0)
        if 3 in self.attack_layer_ids:
            x2_0_pooled = x2_0_pooled + sa3_0.detach() * adv_deltas[3]
            self.assign_sa(sa3_0, self.conv3_0)
        x3_0 = self.conv3_0(x2_0_pooled)
        self.reset_sa(self.conv3_0)

        x3_0_pooled = self.pool(x3_0)
        if 4 in self.attack_layer_ids:
            x3_0_pooled = x3_0_pooled + sa4_0.detach() * adv_deltas[4]
            self.assign_sa(sa4_0, self.conv4_0)
        x4_0 = self.conv4_0(x3_0_pooled)
        self.reset_sa(self.conv4_0)

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

    def get_sa_reserved(self, conv_layer: nn.Sequential):
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
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], attack_layer_ids=[0]):
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

        self.attack_layer_ids = attack_layer_ids

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input):
        with torch.no_grad():
            x0_0 = self.conv0_0(input)
            sa0_0 = self.get_sa_reserved(self.conv0_0)

            x1_0 = self.conv1_0(self.pool(x0_0))
            sa1_0 = self.get_sa_reserved(self.conv1_0)

            x2_0 = self.conv2_0(self.pool(x1_0))
            sa2_0 = self.get_sa_reserved(self.conv2_0)

            x3_0 = self.conv3_0(self.pool(x2_0))
            sa3_0 = self.get_sa_reserved(self.conv3_0)

            x4_0 = self.conv4_0(self.pool(x3_0))
            sa4_0 = self.get_sa_reserved(self.conv4_0)

        if 0 in self.attack_layer_ids:
            input = input + 0.01 * sa0_0.detach() * 2 * (torch.randn(sa0_0.size()).to(sa0_0.device) - 0.5)
            self.assign_sa(sa0_0, self.conv0_0)
        x0_0 = self.conv0_0(input)
        self.reset_sa(self.conv0_0)

        if 1 in self.attack_layer_ids:
            x0_0 = x0_0 + 0.01 * sa1_0.detach() * 2 * (torch.randn(sa1_0.size()).to(sa1_0.device) - 0.5)
            self.assign_sa(sa1_0, self.conv1_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        self.reset_sa(self.conv1_0)

        if 2 in self.attack_layer_ids:
            x1_0 = x1_0 + 0.01 * sa2_0.detach() * 2 * (torch.randn(sa2_0.size()).to(sa2_0.device) - 0.5)
            self.assign_sa(sa2_0, self.conv2_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        self.reset_sa(self.conv2_0)

        if 3 in self.attack_layer_ids:
            x2_0 = x2_0 + 0.01 * sa3_0.detach() * 2 * (torch.randn(sa3_0.size()).to(sa3_0.device) - 0.5)
            self.assign_sa(sa3_0, self.conv3_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        self.reset_sa(self.conv3_0)

        if 4 in self.attack_layer_ids:
            x3_0 = x3_0 + 0.01 * sa4_0.detach() * 2 * (torch.randn(sa4_0.size()).to(sa4_0.device) - 0.5)
            self.assign_sa(sa4_0, self.conv4_0)
        x4_0 = self.conv4_0(self.pool(x3_0))
        self.reset_sa(self.conv4_0)

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
    
    def get_sa_reserved(self, conv_layer: nn.Sequential):
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


class LightWeightNetwork_RA(nn.Module):
    """RA: Random Attacks"""
    def __init__(self, num_classes=1, input_channels=3, block='Res_SP_block', num_blocks=[2,2,2,2], nb_filter=[8, 16, 32, 64, 128], attack_layer_ids=[0]):
        super(LightWeightNetwork_RA, self).__init__()
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

        self.attack_layer_ids = attack_layer_ids

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_train(self, input):
        if 0 in self.attack_layer_ids:
            input = input + 0.01 * 2 * (torch.randn(input.size()).to(input.device) - 0.5)
        x0_0 = self.conv0_0(input)

        if 1 in self.attack_layer_ids:
            x0_0 = x0_0 + 0.01 * 2 * (torch.randn(x0_0.size()).to(x0_0.device) - 0.5)
        x1_0 = self.conv1_0(self.pool(x0_0))

        if 2 in self.attack_layer_ids:
            x1_0 = x1_0 + 0.01 * 2 * (torch.randn(x1_0.size()).to(x1_0.device) - 0.5)
        x2_0 = self.conv2_0(self.pool(x1_0))

        if 3 in self.attack_layer_ids:
            x2_0 = x2_0 + 0.01 * 2 * (torch.randn(x2_0.size()).to(x2_0.device) - 0.5)
        x3_0 = self.conv3_0(self.pool(x2_0))

        if 4 in self.attack_layer_ids:
            x3_0 = x3_0 + 0.01 * 2 * (torch.randn(x3_0.size()).to(x3_0.device) - 0.5)
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
