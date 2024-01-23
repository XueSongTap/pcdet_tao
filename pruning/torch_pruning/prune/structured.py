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

"""Structured pruning."""
import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul
from abc import ABC, abstractstaticmethod
from typing import Sequence, Tuple

# BasePruningFunction 类是一个抽象基类，并且定义了剪枝函数应该实现的通用方法框架。
# 具体来说，它提供了一个类方法 apply，该方法在剪枝过程中被调用，以及两个抽象静态方法 prune_params 和 calc_nparams_to_prune，子类必须实现这些方法以定义如何剪枝参数和计算要剪枝的参数数量。
class BasePruningFunction(ABC):
    """Base pruning function
    """

    @classmethod
    def apply(cls, layer: nn.Module, idxs: Sequence[int], inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
        """Apply the pruning function."""
        idxs = list(set(idxs))
        cls.check(layer, idxs)
        nparams_to_prune = cls.calc_nparams_to_prune(layer, idxs)
        if dry_run:
            return layer, nparams_to_prune
        if not inplace:
            layer = deepcopy(layer)
        layer = cls.prune_params(layer, idxs)
        return layer, nparams_to_prune

    @staticmethod
    def check(layer: nn.Module, idxs: Sequence[int]) -> None:
        """check."""
        pass

    @abstractstaticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        pass

    @abstractstaticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        pass

# ConvPruning 类继承自 BasePruningFunction，用于剪枝卷积层。它重写了 prune_params 和 calc_nparams_to_prune 方法以剪掉特定索引的输出通道。
class ConvPruning(BasePruningFunction):
    """Conv Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune

# GroupConvPruning 类针对分组卷积层，它也重写了 check 方法以确保只支持组数等于输入和输出通道数的层。
class GroupConvPruning(ConvPruning):
    """Group Conv pruning."""

    @staticmethod
    def check(layer, idxs) -> nn.Module:
        """Check."""
        if layer.groups > 1:
            assert layer.groups == layer.in_channels and layer.groups == layer.out_channels, "only group conv with in_channel==groups==out_channels is supported"

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels - len(idxs)
        layer.in_channels = layer.in_channels - len(idxs)
        layer.groups = layer.groups - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

# RelatedConvPruning 类用于剪枝与已有剪枝层相关联的卷积层（例如输入通道）。
class RelatedConvPruning(BasePruningFunction):
    """Related Conv Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        # no bias pruning because it does not change the output size
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[0] * reduce(mul, layer.weight.shape[2:])
        return nparams_to_prune

# LinearPruning 和 RelatedLinearPruning 类似于前面的卷积剪枝类，但它们用于剪枝线性层（全连接层）。
class LinearPruning(BasePruningFunction):
    """Linear Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        layer.out_features = layer.out_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[1] + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune


class RelatedLinearPruning(BasePruningFunction):
    """Related Linear Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[0]
        return nparams_to_prune

#BatchnormPruning 类用于剪枝批量归一化层。
class BatchnormPruning(BasePruningFunction):
    """BatchNorm Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        layer.num_features = layer.num_features - len(idxs)
        layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
        layer.running_var = layer.running_var.data.clone()[keep_idxs]
        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * (2 if layer.affine else 1)
        return nparams_to_prune

# PReLUPruning 类用于剪枝参数化ReLU（PReLU）层。
class PReLUPruning(BasePruningFunction):
    """PReLU pruning."""

    @staticmethod
    def prune_params(layer: nn.PReLU, idxs: list) -> nn.Module:
        """Prune parameters."""
        if layer.num_parameters == 1:
            return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        layer.num_parameters = layer.num_parameters - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.PReLU, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = 0 if layer.num_parameters == 1 else len(idxs)
        return nparams_to_prune


# Funtional
def prune_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune conv."""
    return ConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_related_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune related conv."""
    return RelatedConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_group_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune group conv."""
    return GroupConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_batchnorm(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune Batch Norm."""
    return BatchnormPruning.apply(layer, idxs, inplace, dry_run)


def prune_linear(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune Linear."""
    return LinearPruning.apply(layer, idxs, inplace, dry_run)


def prune_related_linear(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune related linear."""
    return RelatedLinearPruning.apply(layer, idxs, inplace, dry_run)


def prune_prelu(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune prelu."""
    return PReLUPruning.apply(layer, idxs, inplace, dry_run)
