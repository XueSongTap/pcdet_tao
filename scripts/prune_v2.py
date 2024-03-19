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

"""Pruning script for PointPillars."""
import argparse
import datetime
import os
import sys
from pathlib import Path
import tempfile
import torch
import core.loggers.api_logging as status_logging
from core.path_utils import expand_path
import pruning.torch_pruning_v0 as tp
from pcdet.config import (
    cfg, cfg_from_yaml_file
)
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import (
    load_checkpoint,
    load_data_to_gpu
)
from tools.train_utils.train_utils import (
    encrypt_pytorch
)
import inspect
# sys.path.append('/code2/openpcdet_from_tao/')

def parse_args(args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(description="model pruning")
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory.')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pruning_thresh', "-pth", type=float, default=0.1, help='Pruning threshold')
    # parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    return args, cfg
def determine_pruning_strategy(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        # 假设对Conv2d和Linear层默认尝试结构化剪枝
        return "structured"
    elif isinstance(module, torch.nn.BatchNorm2d):
        # BatchNorm层通常随Conv层一起结构化剪枝，单独剪枝时可能更适合非结构化
        return "unstructured"
    else:
        # 对于其他层，默认不剪枝或使用非结构化剪枝
        return "unstructured"

def prune_model():
    """Prune the PointPillars model."""
    args, cfg = parse_args()
    dist_train = False
    args.batch_size = 1
    args.epochs = cfg.train.num_epochs
    threshold = args.pruning_thresh
    if args.output_dir is None:
        if cfg.results_dir is None:
            raise OSError("Either provide results_dir in config file or provide output_dir as a CLI argument")
        else:
            args.output_dir = cfg.results_dir
    args.output_dir = expand_path(args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # Set status logging
    status_file = os.path.join(str(output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars Pruning")
    # -----------------------create dataloader & network & optimizer---------------------------
    train_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.class_names,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs
    )[1]
    input_dict = next(iter(train_loader))
    load_data_to_gpu(input_dict)
    if cfg.prune.model is None:
        raise OSError("Please provide prune.model in config file")
    if not os.path.exists(expand_path(cfg.prune.model)):
        raise OSError(f"Model not found: {cfg.prune.model}")
    model = load_checkpoint(cfg.prune.model)[0]
    model = model.cuda()
    model = model.eval()
    print(model)
    unpruned_total_params = sum(p.numel() for p in model.parameters())
    # strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
    strategy = tp.strategy.L2Strategy()
    DG = tp.DependencyGraph()
    print(input_dict.keys())
    print("inspect.signature(model.forward):", inspect.signature(model.forward))
    DG.build_dependency(model, example_inputs=input_dict)
    # conv layers
    layers = [module for module in model.modules()]
    black_list = layers[-3:]
    for layer in layers:
        if layer in black_list:
            continue
        threshold_run = threshold
        # pruning_idxs
        if isinstance(layer, torch.nn.Conv2d):
            pruning_idxs = strategy(layer.weight, amount=threshold_run)
            pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs)
        elif isinstance(layer, torch.nn.Linear):
            pruning_idxs = strategy(layer.weight, amount=threshold_run)
            pruning_plan = DG.get_pruning_plan(layer, tp.prune_linear, idxs=pruning_idxs)
        else:
            continue
        if pruning_plan is not None:
            pruning_plan.exec()
        else:
             continue
    # conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    # Exclude heads
    # black_conv_layers_list = conv_layers[-3:]
    # count = 0
    # for layer in conv_layers:
    #     if layer in black_conv_layers_list:
    #         continue
    #     # can run some algo here to generate threshold for every node
    #     threshold_run = threshold
    #     pruning_idxs = strategy(layer.weight, amount=threshold_run)
    #     pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs)
    #     if pruning_plan is not None:
    #         pruning_plan.exec()
    #     else:
    #         continue
    #     count += 1

    # linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    # black_linear_layers_list = linear_layers[-3:]
    # for layer in linear_layers:
    #     if layer in black_linear_layers_list:
    #         continue
    #     # Example: Use a different strategy or indices for linear layer pruning
    #     pruning_idxs = strategy(layer.weight, amount=threshold_run)
    #     pruning_plan = DG.get_pruning_plan(layer, tp.prune_linear, idxs=pruning_idxs)
    #     if pruning_plan:
    #         print(pruning_plan)
    #         pruning_plan.exec()
    # batch_normal_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    # black_batch_normal_layers_list = batch_normal_layers[-3:]
    # for layer in batch_normal_layers:
    #     if layer in black_batch_normal_layers_list:
    #         continue
    #     # Example: Use a different strategy or indices for linear layer pruning
    #     pruning_idxs = strategy(layer.weight, amount=threshold_run)
    #     pruning_plan = DG.get_pruning_plan(layer, tp.prune_batchnorm, idxs=pruning_idxs)
    #     if pruning_plan:
    #         print(pruning_plan)
    #         pruning_plan.exec()

    # module_counter = 0  # 初始化模块计数器
    # for name, module in model.named_modules():
    #     if module_counter < 3:
    #         module_counter += 1
    #         continue
    #     strategy_name = determine_pruning_strategy(module)
    #     if strategy_name == "structured":
    #         pruning_idxs = None  # 此处应根据实际情况生成剪枝索引

    #         # 针对Conv2d层的结构化剪枝
    #         if isinstance(module, torch.nn.Conv2d):
    #             pruning_idxs = strategy(module.weight, amount=threshold)  # 生成剪枝索引
    #             pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs)
            
    #         # 针对Linear层的结构化剪枝
    #         elif isinstance(module, torch.nn.Linear):
    #             pruning_idxs = strategy(module.weight, amount=threshold)  # 生成剪枝索引
    #             pruning_plan = DG.get_pruning_plan(module, tp.prune_linear, idxs=pruning_idxs)

    #         # 执行剪枝计划
    #         if pruning_plan:
    #             pruning_plan.exec()
            
            
    #     elif strategy_name == "unstructured":
    #         # 进行非结构化剪枝
    #         if hasattr(module, 'weight'):
    #             weight_mask = torch.rand_like(module.weight) < threshold  # 基于阈值生成随机掩码
    #             module = tp.mask_weight(module, weight_mask)  # 应用权重掩码
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             bias_mask = torch.rand_like(module.bias) < threshold  # 同样方法生成偏置掩码
    #             module = tp.mask_bias(module, bias_mask)  # 应用偏置掩码
    pruned_total_params = sum(p.numel() for p in model.parameters())
    print("Pruning ratio: {}".format(
        pruned_total_params / unpruned_total_params)
    )

    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message="Pruning ratio: {}".format(pruned_total_params / unpruned_total_params)
    )

    save_path = expand_path(f"{args.output_dir}/pruned_{threshold}.pth")
    # handle, temp_file = tempfile.mkstemp()
    # os.close(handle)
    torch.save(model, save_path)
    # encrypt_pytorch(temp_file, save_path, args.key)
    print(f"Pruned model saved to {save_path}")
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message=f"Pruned model saved to {save_path}"
    )
    return model


if __name__ == "__main__":
    try:
        prune_model()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Pruning finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Pruning was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
