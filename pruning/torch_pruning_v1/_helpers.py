import torch.nn as nn
import numpy as np
import torch
from operator import add
from numbers import Number
from collections import namedtuple


#一个命名元组，用于存储未包装的参数和剪枝维度。
UnwrappedParameters = namedtuple('UnwrappedParameters', ['parameters', 'pruning_dim'])

# 一个扩展的命名元组，表示一个依赖组。dep是该组的依赖项，idxs是组内索引的列表。root_idxs是一个占位符，它将被DepGraph填充
class GroupItem(namedtuple('_GroupItem', ['dep', 'idxs'])):
    def __new__(cls, dep, idxs):
        """ A tuple of (dep, idxs) where dep is the dependency of the group, and idxs is the list of indices in the group."""
        cls.root_idxs = None # a placeholder. Will be filled by DepGraph
        return super(GroupItem, cls).__new__(cls, dep, idxs)
    
    def __repr__(self):
        return str( (self.dep, self.idxs) )
# 一个命名元组，表示在当前层和根层中被剪枝维度的索引。idx是当前层中被剪枝维度的索引，root_idx是根层中相应的索引。
class _HybridIndex(namedtuple("_PruingIndex", ["idx", "root_idx"])):
    """ A tuple of (idx, root_idx) where idx is the index of the pruned dimension in the current layer, 
    and root_idx is the index of the pruned dimension in the root layer.
    """
    def __repr__(self):
        return str( (self.idx, self.root_idx) )


# 这两个函数用于将_HybridIndex对象列表转换为仅包含idx或root_idx的普通索引列表。
def to_plain_idxs(idxs: _HybridIndex):
    if len(idxs)==0 or not isinstance(idxs[0], _HybridIndex):
        return idxs
    return [i.idx for i in idxs]

def to_root_idxs(idxs: _HybridIndex):
    if len(idxs)==0 or not isinstance(idxs[0], _HybridIndex):
        return idxs
    return [i.root_idx for i in idxs]

# 判断输入x是否为标量。支持torch.Tensor、数字类型、列表和元组。
def is_scalar(x):
    if isinstance(x, torch.Tensor):
        return len(x.shape) == 0
    elif isinstance(x, Number):
        return True
    elif isinstance(x, (list, tuple)):
        return False
    return False

# 用于处理索引在展平（Flatten）操作中的映射。stride参数控制展平的步长，reverse决定映射方向。
class _FlattenIndexMapping(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs: _HybridIndex):
        new_idxs = []
        
        if self.reverse == True:
            for i in idxs:
                new_idxs.append( _HybridIndex( idx = (i.idx // self._stride), root_idx=i.root_idx ) )
            new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(
                    [ _HybridIndex(idx=k, root_idx=i.root_idx) for k in range(i.idx * self._stride, (i.idx + 1) * self._stride) ]  
                )
        return new_idxs

# 用于处理索引在连接（Concat）操作中的映射。offset定义了连接操作中各部分的偏移量，reverse控制映射方向。
class _ConcatIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: _HybridIndex):

        if self.reverse == True:
            new_idxs = [
                _HybridIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx )
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        else:
            new_idxs = [ _HybridIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        return new_idxs

# 用于处理索引在分割（Split）操作中的映射。与_ConcatIndexMapping类似，但用于分割操作。
class _SplitIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: _HybridIndex):
        if self.reverse == True:
            new_idxs = [ _HybridIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        else:
            new_idxs = [
                _HybridIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx)
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        return new_idxs

# ScalarSum 和 VectorSum:
# 这两个类用于在剪枝过程中累计标量和向量（列表或torch.Tensor）类型的度量。update方法用于更新特定度量的值，results返回当前所有度量的累计结果，reset重置内部状态。
class ScalarSum:
    def __init__(self):
        self._results = {}

    def update(self, metric_name, metric_value):
        if metric_name not in self._results:
            self._results[metric_name] = 0
        self._results[metric_name] += metric_value

    def results(self):
        return self._results

    def reset(self):
        self._results = {}


class VectorSum:
    def __init__(self):
        self._results = {}

    def update(self, metric_name, metric_value):
        if metric_name not in self._results:
            self._results[metric_name] = metric_value
        if isinstance(metric_value, torch.Tensor):
            self._results[metric_name] += metric_value
        elif isinstance(metric_value, list):
            self._results[metric_name] = list(
                map(add, self._results[metric_name], metric_value)
            )

    def results(self):
        return self._results

    def reset(self):
        self._results = {}
