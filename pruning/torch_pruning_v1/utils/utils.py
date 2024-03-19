from ..ops import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR
from ..ops import module2type
import torch
from .op_counter import count_ops_and_params
import torch.nn as nn


# @torch.no_grad() 装饰器：应用于count_params函数，
# 以防止PyTorch跟踪用于梯度计算的操作，因为计算参数数量时不需要这些操作。
@torch.no_grad()
def count_params(module):
    # 通过总结每个参数张量中的元素数量，计算给定PyTorch模块（例如，神经网络层或整个模型）中的参数总数
    return sum([p.numel() for p in module.parameters()])
# 递归地将输入对象（张量、列表、元组和字典）平铺成一个单一的列表。这个实用程序可以用于处理以各种嵌套结构出现的模型输出或中间表示。
# def flatten_as_list(obj):
#     # 这个函数将输入对象（可以是张量、列表、元组或字典）递归地平铺成一个列表。如果输入是张量，则直接返回包含该张量的列表；如果是列表或元组，递归平铺其元素；如果是字典，递归平铺其值；否则，直接返回输入对象。
#     if isinstance(obj, torch.Tensor):
#         return [obj]
#     elif isinstance(obj, (list, tuple)):
#         flattened_list = []
#         for sub_obj in obj:
#             flattened_list.extend(flatten_as_list(sub_obj))
#         return flattened_list
#     elif isinstance(obj, dict):
#         flattened_list = []
#         for sub_obj in obj.values():
#             flattened_list.extend(flatten_as_list(sub_obj))
#         return flattened_list
#     else:
#         return obj


def flatten_as_list(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        flattened_list = []
        for sub_obj in obj:
            flattened_list.extend(flatten_as_list(sub_obj))
        return flattened_list
    elif isinstance(obj, dict):
        flattened_list = []
        for sub_obj in obj.values():
            flattened_list.extend(flatten_as_list(sub_obj))
        return flattened_list
    elif isinstance(obj, int):
        return [obj]
    else:
        return obj
# 可视化函数（draw_computational_graph、draw_groups、draw_dependency_graph）：这些函数创建不同方面的神经网络结构和模块之间依赖关系的可视化表示：

# draw_computational_graph：绘制计算图，展示不同模块（节点）如何连接。
# draw_groups：基于某些标准可视化模块组，很可能与修剪对它们的影响有关，指示被一起考虑修剪的紧密连接的模块。
# draw_dependency_graph：更详细地展示模块之间的依赖关系，包括影响方向，可能还区分不同类型的连接（例如，修剪依赖）。
def draw_computational_graph(DG, save_as, title='Computational Graph', figsize=(16, 16), dpi=200, cmap=None):
    """
    DG: 依赖图对象，存储了模型中模块之间的依赖关系。
    save_as: 图片保存的路径。
    title: 图像的标题，默认为'Computational Graph'。
    figsize: 图像的大小，默认为(16, 16)。
    dpi: 图像的分辨率，默认为200。
    cmap: 颜色映射，默认使用matplotlib的'Blues'颜色映射。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = len(DG.module2node)
    # 函数通过DG.module2node获取模型中所有模块的节点表示。
    module2idx = {m: i for (i, m) in enumerate(DG.module2node.keys())}
    # 创建一个全0的方阵G，其尺寸等于节点数的平方，用于表示节点间的连接情况。
    G = np.zeros((n_nodes, n_nodes)) # 可视化函数使用矩阵（numpy.zeros）来表示图中节点之间的连接。这些矩阵根据DependencyGraph（可能由DG参数表示）对象中定义的关系来填充，该对象跟踪模块、它们的连接以及额外属性，如修剪函数。
    fill_value = 1
    # 遍历每个模块及其对应的节点，根据节点的输入和输出关系填充矩阵G，其中矩阵的值被设为填充值（通常为1），表示存在连接。
    for module, node in DG.module2node.items():
        for input_node in node.inputs:
            G[module2idx[input_node.module], module2idx[node.module]] = fill_value
            G[module2idx[node.module], module2idx[input_node.module]] = fill_value
        for out_node in node.outputs:
            G[module2idx[out_node.module], module2idx[node.module]] = fill_value
            G[module2idx[node.module], module2idx[out_node.module]] = fill_value
        pruner = DG.get_pruner_of_module(module)
    fig, ax = plt.subplots(figsize=(figsize))
    # 使用matplotlib创建一个图形和轴，然后用imshow函数将矩阵G绘制为图像，颜色映射由cmap参数指定。
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
    # plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax

# 此函数专注于可视化模型中的分组信息，这对于理解哪些模块在修剪过程中被视为一个整体特别有用。
def draw_groups(DG, save_as, title='Group', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = 2*len(DG.module2node)
    node2idx = {m: i for (i, m) in enumerate(DG.module2node.values())}
    G = np.zeros((n_nodes, n_nodes))
    fill_value = 10
    # 代码中提到了“修剪”操作，这是通过移除不那么重要的连接或神经元来减小神经网络大小的技术。这反映在函数和方法如get_pruner_of_module、prune_out_channels、get_out_channels以及get_pruning_group中，表明该模块还支持分析修剪对网络结构影响的功能。
    for i, (module, node) in enumerate(DG.module2node.items()):
        pruning_fn = DG.get_pruner_of_module(module).prune_out_channels
        prunable_ch = DG.get_out_channels(module)
        if prunable_ch is None: continue
        # 实现步骤与draw_computational_graph大体相似，但关注点在于模块的分组而非单纯的连接关系。
        # 核心差异在于它通过DG.get_pruning_group获取与修剪相关的分组信息，然后根据这些分组信息填充矩阵G。
        group = DG.get_pruning_group(module, pruning_fn, list(range(prunable_ch)))
        grouped_idxs = []
        for dep, _ in group:
            source, target, trigger, handler = dep.source, dep.target, dep.trigger, dep.handler
            if DG.is_out_channel_pruning_fn(trigger):
                grouped_idxs.append(node2idx[source]*2+1)
            else:
                grouped_idxs.append(node2idx[source]*2)

            if DG.is_out_channel_pruning_fn(handler):
                grouped_idxs.append(node2idx[target]*2+1)
            else:
                grouped_idxs.append(node2idx[target]*2)
        grouped_idxs = list(set(grouped_idxs))
        for k1 in grouped_idxs:
            for k2 in grouped_idxs:
                G[k1, k2] = fill_value

    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#999999", linewidth=0.1)
    # plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#999999", linewidth=0.1)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax


def draw_dependency_graph(DG, save_as, title='Group', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = len(DG.module2node)
    node2idx = {node: i for (i, node) in enumerate(DG.module2node.values())}
    # 类似于draw_computational_graph，但它创建了一个更复杂的矩阵G，以反映模块之间依赖性的方向和类型。
    G = np.zeros((2*n_nodes, 2*n_nodes))
    fill_value = 10
    #     遍历每个模块及其依赖关系，根据依赖的源模块和目标模块以及触发器和处理器的类型，填充矩阵G。
    # 如果修剪函数影响了模块的输入和输出通道，这些关系将在矩阵中得到反映，显示为不同模块间的依赖路径。
    for module, node in DG.module2node.items():
        for dep in node.dependencies:
            trigger = dep.trigger
            handler = dep.handler
            source = dep.source
            target = dep.target

            if DG.is_out_channel_pruning_fn(trigger):
                G[2*node2idx[source]+1, 2*node2idx[target]] = fill_value
            else:
                G[2*node2idx[source], 2*node2idx[target]+1] = fill_value

        pruner = DG.get_pruner_of_module(module)
        if pruner.prune_out_channels == pruner.prune_in_channels:
            G[2*node2idx[node], 2*node2idx[node]+1] = fill_value

    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, 2*n_nodes)+0.5, xmin=np.full(2*n_nodes, 0)-0.5, xmax=np.full(2*n_nodes, 2*n_nodes)-0.5, color="#999999", linewidth=0.05)
    # plt.vlines(x=np.arange(0, 2*n_nodes)+0.5, ymin=np.full(2*n_nodes, 0)-0.5, ymax=np.full(2*n_nodes, 2*n_nodes)-0.5, color="#999999", linewidth=0.05)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax