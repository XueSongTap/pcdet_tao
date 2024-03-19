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
import pruning.torch_pruning_v1 as tp
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
# def custom_forward_fn(model, example_inputs):
#     # 假设example_inputs已经是正确格式的字典
#     # 直接将其传递给模型的forward方法
#     output_dict = model(**example_inputs)
    
#     # 如果需要，可以从output_dict中提取必要的信息进行后续处理
#     # 例如，如果后续步骤只需要'points'数据：
#     points_output = output_dict['points']
    
#     # 根据需要返回整个字典或其特定部分
#     return output_dict  # 或 return points_output


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
    print("model:", model)
    model = model.cuda()
    # model = model.eval()
    unpruned_total_params = sum(p.numel() for p in model.parameters())
    # strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
    # strategy = tp.strategy.L2Strategy()
    importance_strategy = tp.importance.GroupNormImportance(p=2, group_reduction="mean")
    DG = tp.DependencyGraph()
    # print("example_inputs:", input_dict)
    input_dict = {'batch_dict': input_dict}

    print(input_dict.keys())
    print("inspect.signature(model.forward):", inspect.signature(model.forward))
    DG.build_dependency(model, example_inputs=input_dict)

    # DG.build_dependency(model, example_inputs=input_dict, forward_fn=custom_forward_fn)
    # conv layers
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    # Exclude heads
    black_list = layers[-3:]
    count = 0
    for layer in layers:
        group = DG.get_pruning_group(layer)
        imp_score = importance_strategy(group)
        # 基于重要性评分和剪枝阈值来选择剪枝的索引
        threshold_run = threshold
        pruning_idxs = imp_score < threshold_run
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs)
        if pruning_plan is not None:
            pruning_plan.exec()
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
