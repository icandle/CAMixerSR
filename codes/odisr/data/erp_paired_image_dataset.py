import math
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .utils import paired_random_crop
import numpy as np
import cv2
import torch
import os.path as osp


@DATASET_REGISTRY.register()
class ERPPairedImageDataset(data.Dataset):
    """Paired image dataset for conditional ODI restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ERPPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if 'ext_dataroot_gt' in self.opt:
            assert self.io_backend_opt['type'] == 'disk'
            self.ext_gt_folder, self.ext_lq_folder = opt['ext_dataroot_gt'], opt['ext_dataroot_lq']
            if 'enlarge_scale' in self.opt:
                enlarge_scale = self.opt['enlarge_scale']
            else:
                enlarge_scale = [1 for _ in range(len(self.ext_gt_folder)+1)]

            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl) \
                         * enlarge_scale[0]
            for i in range(len(self.ext_gt_folder)):
                self.paths += paired_paths_from_folder([self.ext_lq_folder[i], self.ext_gt_folder[i]], ['lq', 'gt'],
                                                          self.filename_tmpl) * enlarge_scale[i+1]
        else:
            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                              self.opt['meta_info_file'], self.filename_tmpl)
            else:
                self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        if 'gt_size' in self.opt and self.opt['gt_size']:
            self.glob_condition = get_condition(self.opt['gt_h']//self.opt['scale'],
                                                self.opt['gt_w']//self.opt['scale'], self.opt['condition_type'])

        if 'sub_image' in self.opt and self.opt['sub_image']:
            self.sub_image = True
        else:
            self.sub_image = False


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        if self.sub_image:
            sub_h, sub_w = osp.split(lq_path)[-1].split('_')[3:5]
            sub_h, sub_w = int(sub_h) // scale, int(sub_w) // scale
        else:
            sub_h, sub_w = 0, 0

        if self.opt.get('force_resize'):
        # resize gt with wrong resolutions
            img_gt = cv2.resize(img_gt, (img_lq.shape[1] * scale, img_lq.shape[0] * scale), cv2.INTER_CUBIC)

        # augmentation for training
        # random crop
        if 'gt_size' in self.opt and self.opt['gt_size']:
            gt_size = self.opt['gt_size']
            img_gt, img_lq, top_lq, left_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path,
                                                                       return_top_left=True)
            top_lq, left_lq = top_lq + sub_h, left_lq + sub_w
            if self.opt['condition_type'] is not None:
                if ('DIV2K' or 'Flickr2K') in lq_path:
                    _condition = torch.zeros([1, img_lq.shape[0], img_lq.shape[1]])
                else:
                    _condition = self.glob_condition[:,top_lq:top_lq+img_lq.shape[0],left_lq:left_lq+img_lq.shape[1]]
            else:
                _condition = 0.
        else:
            _condition = get_condition(img_lq.shape[0], img_lq.shape[1], self.opt['condition_type'])
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'condition': _condition}

    def __len__(self):
        return len(self.paths)



def get_condition(h, w, condition_type):
    if condition_type is None:
        return 0.
    elif condition_type == 'cos_latitude':
        return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)
    elif condition_type == 'latitude':
        return make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2, 0, 1) * math.pi / 2
    elif condition_type == 'coord':
        return make_coord([h, w]).permute(2, 0, 1)
    else:
        raise RuntimeError('Unsupported condition type')


def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

