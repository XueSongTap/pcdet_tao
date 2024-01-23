# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unstructured pruning."""
import torch
from copy import deepcopy


__all__ = ['mask_weight', 'mask_bias']


def _mask_weight_hook(module, inp):
    if hasattr(module, 'weight_mask'):
        module.weight.data *= module.weight_mask


def _mask_bias_hook(module, inp):
    if module.bias is not None and hasattr(module, 'bias_mask'):
        module.bias.data *= module.bias_mask

"""
mask_weight 函数的目的是将权重矩阵与一个掩码相乘，掩码中的元素为0或1。其中1表示保留相应的权重，而0表示在前向传播过程中将相应的权重设置为0（剪枝）。该函数接受三个参数：要剪枝的层（layer）、掩码（mask）和一个布尔值inplace，表示是否在原地修改层或返回一个副本。
"""
def mask_weight(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if mask.shape != layer.weight.shape:
        return layer
    mask = torch.tensor(mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False)
    if hasattr(layer, 'weight_mask'):
        mask = mask + layer.weight_mask
        mask[mask > 0] = 1
        layer.weight_mask = mask
    else:
        layer.register_buffer('weight_mask', mask)

    layer.register_forward_pre_hook(_mask_weight_hook)
    return layer

"""
mask_bias 函数的工作方式类似于mask_weight，但是它针对的是层的偏置项。

这两个函数都进行了以下步骤：

如果inplace参数为False，则创建层的一个深拷贝。

检查掩码的形状是否与权重或偏置的形状相匹配。

将掩码转换为合适的数据类型和设备，并创建一个不需要梯度（requires_grad=False）的tensor。

如果层已经有一个掩码，新的掩码将与旧的掩码相加，然后掩码中大于0的元素设置为1。

如果层没有掩码，则使用register_buffer方法将掩码添加到层的状态中。这样可以确保掩码与层的权重或偏置保持在相同的设备上，并且在模型保存和加载时掩码也会一起被处理。

使用register_forward_pre_hook注册一个钩子函数，该函数会在每次前向传播之前应用掩码。
"""
def mask_bias(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if layer.bias is None or mask.shape != layer.bias.shape:
        return layer

    mask = torch.tensor(mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False)
    if hasattr(layer, 'bias_mask'):
        mask = mask + layer.bias_mask
        mask[mask > 0] = 1
        layer.bias_mask = mask
    else:
        layer.register_buffer('bias_mask', mask)
    layer.register_forward_pre_hook(_mask_bias_hook)
    return layer
