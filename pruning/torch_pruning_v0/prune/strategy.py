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

"""Strategy of pruning."""
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random




def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """这个函数用于根据指定的round_to值调整剪枝数量，保证剪枝后的参数数量是round_to的整数倍。
    """
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """ 
    round_to = int(round_to)
    if round_to <= 1:
        return n_to_prune
    after_pruning = total_parameters - n_to_prune
    compensation = after_pruning % round_to
    # round to the nearest (round_to * N)
    # avoid negative n_to_prune
    if (compensation < round_to // 2 and after_pruning > round_to) or round_to > n_to_prune:
        n_to_prune = n_to_prune + compensation  # floor
    else:
        n_to_prune = n_to_prune - round_to + compensation  # ceiling
    return n_to_prune


class BaseStrategy(ABC):
    """Base Strategy class."""
    """这是一个抽象基类（ABC），它定义了所有剪枝策略应该遵循的接口。它要求子类实现apply方法，该方法根据用户指定的剪枝百分比来应用策略。"""
    def __call__(self, *args, **kwargs):
        """Call method."""
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(cls, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0)
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError


class RandomStrategy(BaseStrategy):
    """Random Strategy class."""
    """这个类继承自BaseStrategy，实现了随机剪枝策略。它根据指定的剪枝百分比amount随机选择权重进行剪枝。
    """
    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """Apply the strategy."""
        if amount <= 0:
            return []
        n = len(weights)
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        indices = random.sample(list(range(n)), k=n_to_prune)
        return indices

"""
L1范数和L2范数都是用来度量向量长度或大小的方法，但它们计算的方式和具有的属性有所不同。
L1范数（也称为曼哈顿距离或L1距离）:
L1范数是向量元素绝对值之和。
对于向量x，L1范数可以表示为：||x||_1 = Σ|xi|。
L1范数对异常值不太敏感，因此在处理含有异常值的数据时常用L1正则化。
在优化问题中，使用L1范数作为正则化项可以导致稀疏解，即很多参数会变成零。这对于特征选择或者高维数据（其中很多特征可能是无关紧要的）是非常有用的。
L2范数（也称为欧几里得距离或L2距离）:
L2范数是向量元素平方和的平方根。
对于向量x，L2范数可以表示为：||x||_2 = √(Σxi^2)。
L2范数对异常值很敏感，因为平方项放大了异常值的影响。
在优化问题中，使用L2范数作为正则化项可以防止参数值变得过大，导致过拟合。这种正则化通常被称为岭回归或Tikhonov正则化。
在上下文中，L1和L2范数被用作剪枝策略的基准。在L1剪枝策略中，参数向量的L1范数（参数的绝对值之和）用来度量参数的重要性；在L2剪枝策略中，使用L2范数（参数的平方和的平方根）来度量。通常，参数的范数越小，被认为越不重要，因此在剪枝过程中更有可能被移除。
"""
class LNStrategy(BaseStrategy):
    """LN magnitude based pruning strategy.

    Two mode of LN-magnitude-based (L1 or L2) pruning startegy are provided through this class:
    - "amount": The pruning algorithm in original Torch-pruning. "amount" means the ratio of
    number of filters to be pruned to the total number of filters. Suppose the total number of
    filters is N, then the number of filters to be pruned is N * amount. The filters are sorted
    along the LN-magnitude of each filter and the smallest N* amount filters will be pruned.
    - "thresh": The pruning algorithm in tao-keras. The filter with smaller LN-magnitude than
    a threshold will be pruned.
    这个类同样继承自BaseStrategy，实现了基于L1或L2范数的剪枝策略。它支持两种模式：amount（剪枝比例）和thresh（阈值）。根据范数值对权重进行排序，然后根据模式移除最小的或低于阈值的权重。
    Common tricks:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def __init__(self, p, mode="amount"):
        """Constructor for LNS strategy."""
        self.p = p
        self.mode = mode
        if self.mode not in ["amount", "thresh"]:
            raise ValueError("Only support \"amount\" and \"thresh\" mode")

    def apply(self, weights, amount=0.0, round_to=1, scores=None) -> Sequence[int]:  # return index
        """Apply the pruning."""
        if amount <= 0:
            return []
        n = len(weights)
        if scores is None:
            l1_norm = torch.norm(weights.view(n, -1), p=self.p, dim=1)
        else:
            l1_norm = scores

        if self.mode == "amount":
            n_to_prune = int(amount * n) if amount < 1.0 else amount
            n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
            if n_to_prune == 0:
                return []
            threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
            indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        elif self.mode == "thresh":
            # Thresh is the strategy in tao-tf
            l1_norm /= torch.max(l1_norm)
            remained_idx = torch.nonzero(l1_norm > amount).view(-1).tolist()
            num_remained = len(remained_idx)
            # Granularity
            if num_remained % round_to > 0:
                num_remained += round_to - (num_remained % round_to)
            num_remained = min(num_remained, n)
            if num_remained == n:
                return []
            sorted_idx = torch.argsort(-l1_norm)
            indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist()

        return indices

"""
给定的代码片段定义了一个名为CustomScoreStrategy的类，它继承自BaseStrategy。这个类提供了一个名为apply的方法，根据给定的阈值和粒度对一组分数进行剪枝。
"""
class CustomScoreStrategy(BaseStrategy):
    """Custom Score Strategy.

    A helper class to execute sorting and filtering with any pruning score.

    common trick:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def apply(self, scores, thresh=0.0, round_to=1) -> Sequence[int]:
        """Apply the pruning."""
        """apply方法接受分数列表、阈值（thresh）和粒度（round_to）作为参数。它返回一个通过剪枝的索引序列。"""
        # 方法首先检查阈值是否小于等于0。如果是，则返回一个空列表，表示不需要进行剪枝。
        if thresh <= 0:
            return []
        n = len(scores) # 然后确定分数列表的长度（n），并使用torch.nonzero(scores > thresh)找到大于阈值的分数的索引。结果索引被展平并转换为Python列表（remained_idx）。
        remained_idx = torch.nonzero(scores > thresh).view(-1).tolist() # 剩余索引的数量（num_remained）根据remained_idx的长度计算得出。
        num_remained = len(remained_idx) # 代码通过检查num_remained是否不能被round_to整除来应用粒度。如果不能整除，则将num_remained增加到下一个round_to的倍数。
        # Granularity 代码通过将num_remained与round_to进行比较，并取最大值，确保至少保留round_to个索引。
        if num_remained % round_to > 0:
            num_remained += round_to - (num_remained % round_to)
        # keep the min idxs 
        num_remained = max(num_remained, round_to) # 代码通过将num_remained与round_to进行比较，并取最大值，确保至少保留round_to个索引。
        num_remained = min(num_remained, n) # 代码还通过将num_remained与n进行比较，并取最小值，确保剩余索引的数量不超过分数列表的长度。
        # 如果剩余索引的数量等于分数列表的长度（num_remained == n），表示不需要进行剪枝，将返回一个空列表。
        if num_remained == n:
            return []
        sorted_idx = torch.argsort(-scores) # 否则，使用torch.argsort(-scores)对分数进行降序排序，并选择与前num_remained个分数对应的索引。
        indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist() # 最后，使用torch.sort对选择的索引进行升序排序，并将其转换为Python列表（indices），然后返回。

        return indices


class L1Strategy(LNStrategy):
    """L1 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L1Strategy, self).__init__(p=1)


class L2Strategy(LNStrategy):
    """L2 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L2Strategy, self).__init__(p=2)
