
"""Inference script for PointPillars."""
import argparse
import datetime
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

try:
    import tensorrt as trt  # pylint: disable=unused-import  # noqa: F401
    from tools.export.tensorrt_model import TrtModel
except:  # noqa: E722
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, "
        "inference with TensorRT engine will not be available."
    )
import core.loggers.api_logging as status_logging
from core.path_utils import expand_path
from tools.eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import load_checkpoint
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

# parse_config函数用于解析命令行参数并从YAML配置文件加载配置到全局配置对象cfg中。
def parse_config():
    """Argument Parser."""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="output checkpoint directory.")
    parser.add_argument(
        "--trt_engine",
        type=str,
        required=False,
        default=None,
        help="Path to the TensorRT engine to be used for inference"
    )
    # parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    np.random.seed(1024)
    return args, cfg

#parse_epoch_num函数用于从模型文件名中解析出训练的epoch数。
def parse_epoch_num(model_file):
    """Parse epoch number from model file."""
    model_base = os.path.basename(model_file)
    epoch_string = model_base[:-4].split("_")[-1]
    return int(epoch_string)

# infer_single_ckpt函数用于使用PyTorch模型进行一次推理。
def infer_single_ckpt(
    model, test_loader, args,
    infer_output_dir, logger,
    cfg
):
    start_time = time.time()  # 开始计时

    """Do inference with PyTorch model."""
    model.cuda()
    eval_utils.infer_one_epoch(
        cfg, model, test_loader, logger,
        result_dir=infer_output_dir, save_to_file=args.save_to_file
    )
    end_time = time.time()  # 结束计时
    logger.info(f"Inference time: {end_time - start_time} seconds")  # 输出推理耗时
    

# infer_single_ckpt_trt函数用于使用TensorRT引擎进行一次推理。
def infer_single_ckpt_trt(
    model, test_loader, args,
    infer_output_dir, logger,
    cfg
):
    """Do inference with TensorRT engine."""
    eval_utils.infer_one_epoch_trt(
        cfg, model, test_loader, logger,
        result_dir=infer_output_dir, save_to_file=args.save_to_file
    )
# CustomNMS是一个自定义的非最大抑制（NMS）模块，在推理过程中用于去除冗余的边界框。
class CustomNMS(nn.Module):
    """Customized NMS module."""

    def __init__(self, post_process_cfg):
        """Initialize."""
        super().__init__()
        self.post_process_cfg = post_process_cfg

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        batch_output = []
        for idx, box_per_frame in enumerate(output_boxes):
            num_box_per_frame = num_boxes[idx]
            box_per_frame = torch.from_numpy(box_per_frame).cuda()
            box_per_frame = box_per_frame[:num_box_per_frame, ...]
            box_preds = box_per_frame[:, 0:7]
            label_preds = box_per_frame[:, 7] + 1
            cls_preds = box_per_frame[:, 8]
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=self.post_process_cfg.nms_config,
                score_thresh=self.post_process_cfg.score_thresh
            )
            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
            final_output = torch.cat(
                [
                    final_boxes,
                    final_scores.view((-1, 1)),
                    final_labels.view((-1, 1))
                ],
                axis=-1
            )
            batch_output.append(final_output.cpu().numpy())
        return batch_output

# CustomPostProcessing是一个自定义的后处理模块，它使用CustomNMS来处理推理结果。
class CustomPostProcessing(nn.Module):
    """Customized PostProcessing module."""

    def __init__(self, model, cfg):
        """Initialize."""
        super().__init__()
        self.model = model
        self.custom_nms = CustomNMS(cfg)

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        return self.custom_nms(
            output_boxes,
            num_boxes
        )


class TrtModelWrapper():
    """TensorRT engine wrapper."""

    def __init__(self, model, cfg, trt_model):
        """Initialize."""
        self.model = model
        self.cfg = cfg
        self.trt_model = trt_model
        self.post_processor = CustomPostProcessing(
            self.model,
            self.cfg.model.post_processing
        )

    def __call__(self, input_dict):
        """Call method."""
        trt_output = self.trt_model.predict(input_dict)
        return self.post_processor(
            trt_output["output_boxes"],
            trt_output["num_boxes"],
        )
"""
main函数是脚本的入口点。它执行以下步骤：

解析命令行参数和配置文件。
设置输出目录和日志记录器。
创建数据加载器。
加载模型和TensorRT引擎（如果提供了）。
进行推理，并将结果保存到指定的输出目录中。
"""

def main():
    """Main function."""
    args, cfg = parse_config()
    args.batch_size = cfg.inference.batch_size
    args.workers = cfg.dataset.num_workers
    args.ckpt = cfg.inference.checkpoint
    if args.output_dir is None:
        if cfg.results_dir is None:
            raise OSError("Either provide results_dir in config file or provide output_dir as a CLI argument")
        else:
            args.output_dir = cfg.results_dir
    output_dir = Path(expand_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_output_dir = output_dir / 'infer'
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = infer_output_dir / ('log_infer_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    # log to file
    logger.info('**********************Start logging**********************')
    # Set status logging
    status_file = os.path.join(str(infer_output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars inference")
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.class_names,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )[1]
    model = load_checkpoint(
        args.ckpt
        # args.key
    )[0]
    # Try to load TRT engine if there is any
    if args.trt_engine is not None:
        trt_model = TrtModel(
            args.trt_engine,
            args.batch_size,
        )
        trt_model.build_or_load_trt_engine()
        # Check the batch size
        engine_batch_size = trt_model.engine._engine.get_binding_shape(0)[0]
        if engine_batch_size != args.batch_size:
            raise ValueError(f"TensorRT engine batch size: {engine_batch_size}, mismatch with "
                             f"batch size for evaluation: {args.batch_size}. "
                             "Please make sure they are the same by generating a new engine or "
                             f"modifying the evaluation batch size in spec file to {engine_batch_size}.")
        model_wrapper = TrtModelWrapper(
            model,
            cfg,
            trt_model
        )
        with torch.no_grad():
            infer_single_ckpt_trt(
                model_wrapper, test_loader, args,
                infer_output_dir, logger, cfg
            )
    else:
        # Load model from checkpoint
        with torch.no_grad():
            infer_single_ckpt(
                model, test_loader, args, infer_output_dir,
                logger, cfg
            )


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
"""
TensorRT特定优化：

Layer Fusion：将多个层合并为一个层，减少内存访问和增加执行效率。
Kernel Auto-Tuning：TensorRT会为特定的操作选择最优的CUDA内核。
Dynamic Tensor Memory (DTM)：优化内存分配和管理，减少内存使用和数据移动。
使用官方或定制的TensorRT插件来实现特殊层或操作。
推理设置：

执行批量推理：使用较大的批次大小来减少调用模型的次数。
使用流水线处理：将数据传输和计算分开，同时执行以减少延迟。



TensorRT特定优化和推理设置的大部分都是在构建TensorRT引擎（也就是在执行export操作）的过程中进行的，但是有些调整和优化可以在引擎构建后执行，尤其是与推理执行相关的设置。

Layer Fusion：

在构建TensorRT引擎时，会自动尝试合并层以提高执行效率。
Kernel Auto-Tuning：

同样是在构建引擎时，TensorRT会基于硬件配置进行内核的自动调优。
Dynamic Tensor Memory (DTM)：

DTM是TensorRT用于优化内存分配的技术，通常在构建引擎时处理。
使用官方或定制的TensorRT插件：

如果模型中有TensorRT不支持的操作，可以在构建引擎时添加自定义插件。
在export脚本中，可以通过设置相关的参数和调用相应的API来实现上述优化。例如，可以通过设置不同的精度（如FP16或INT8），或者通过设置工作空间的大小来影响内核的选择和层的融合。

执行批量推理：

批量大小可以在构建TensorRT引擎时指定，通常作为导出脚本中的一个参数。
引擎构建完成后，在推理时，可以按照指定的批量大小来提供输入数据。
使用流水线处理：

这与具体如何部署和执行引擎有关，可以通过多线程或异步执行API来实现。
在推理时，可以将数据的预处理、模型推理和后处理步骤分开并行处理。
总结来说，大多数TensorRT优化是在构建引擎的过程中进行的，此时引擎会针对指定的硬件和设置进行优化。一旦引擎构建完成并序列化保存，就不再修改这些优化设置。但是，推理时的执行策略，如批量大小和流水线处理，则可以在实际运行推理时根据需要进行调整。



在寻找模型推理性能瓶颈并进行优化时，您可以按照以下步骤进行：

性能分析：

使用性能分析工具：NVIDIA提供了Nsight Systems、Nsight Compute和TensorRT的内置分析器来帮助诊断性能问题。
运行性能分析来收集推理过程中的时间和资源消耗数据。观察各层耗时、内存使用情况和硬件利用率。
识别瓶颈：

确定推理时间中的高耗时操作或层。
检查是否有大量的内存传输或数据复制操作。
观察GPU利用率，检查是否存在GPU空闲的情况。
优化策略选择：

对于耗时的操作，考虑合并计算密集型的层（Layer Fusion）或使用更高效的算子。
针对内存瓶颈，减少数据移动或使用更高效的内存管理策略（如Dynamic Tensor Memory）。
如果GPU利用率低，考虑增加批量大小或使用流水线来更充分地利用GPU资源。
实施优化：

调整TensorRT构建参数：例如，更改工作空间大小、启用层融合、使用动态形状等。
使用FP16或INT8精度减少计算负载，但要注意验证模型的准确度。
如果模型中存在TensorRT不支持的定制层，考虑实现自定义插件。
实施批量推理和流水线处理以提高推理吞吐量。
再次性能分析：

在进行了优化之后，再次使用性能分析工具来评估优化的效果。
比较优化前后的性能指标，如推理时间、资源消耗和准确度。
迭代调优：

推理性能优化是一个迭代的过程，可能需要多次调整和测试。
持续监控新版本的TensorRT和CUDA库，因为它们可能带来新的优化功能和性能提升。
实际部署考虑：

在优化时，考虑实际部署环境（服务器、边缘设备等）的硬件资源和限制。
优化推理延迟和吞吐量时，要根据目标应用场景的需求进行平衡。
在整个优化过程中，请确保跟踪重要的性能指标，并验证优化没有对模型的精确度产生负面影响。通常，性能优化和模型精确度之间需要做出权衡。使用适当的工具和方法，结合对模型和部署环境的深入了解，可以有效地识别和解决推理性能瓶颈。


"""