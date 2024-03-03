import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_metric.py')]
_model_modules = [importlib.import_module(f'odisr.metrics.{file_name}') for file_name in loss_filenames]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
