import numpy as np
import torch
import torch.nn.functional as F
import math
from PIL import Image
from torchvision import transforms


def make_coord(shape, ranges=None, flatten=True, double=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    if double:
        return ret.double()
    return ret


def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0


def erp_downsample_fisheye(erp_hr: torch.Tensor, scale, downsample_type='bicubic',
                           fisheye_expand_scale=1.25, fisheye_patch_scale=1.5):
    """
    backward (implicit bicubic downsample process based on uniformed dual fisheye)
    :param fisheye_patch_scale: float, fisheye resolution [h * scale, h * scale]
                                1.5 is recommended to maintain a consistent average pixel density on Fisheye and ERP
    :param fisheye_expand_scale: expand 180 degree fisheye for edge consistency
    :param erp_hr: (Tensor) [c, h, w]
    :param scale: downsample scale
    :param downsample_type: str, bicubic or bilinear
    :return:
        erp_lr (Tensor): lr Tensor [c, h, w] in [0, 1]
    """
    h, w = erp_hr.shape[-2:]
    erp_hr = erp_hr[:, :h // scale * scale, :w // scale * scale]

    fisheye_patch_size = round(h * fisheye_patch_scale * fisheye_expand_scale)

    # fisheye
    fisheye_rgb = torch.zeros([2, 3, fisheye_patch_size, fisheye_patch_size])
    fisheye_coord = make_coord([*fisheye_rgb.shape[-2:]], flatten=False)
    fisheye_rho = torch.sqrt(fisheye_coord[:, :, 0] ** 2 + fisheye_coord[:, :, 1] ** 2)
    fisheye_rho *= fisheye_expand_scale
    fisheye_theta = torch.arctan2(fisheye_coord[:, :, 0], fisheye_coord[:, :, 1])
    x_fisheye, y_fisheye = np.where(fisheye_rho.numpy() <= fisheye_expand_scale)
    # expand z-dimension
    rho = fisheye_rho[y_fisheye, x_fisheye]
    theta = fisheye_theta[y_fisheye, x_fisheye]
    x_erp = theta / math.pi
    y_erp = 1 - rho

    for i in [-1, 1]:
        erp_coord = torch.stack([x_erp, y_erp * i], dim=-1)
        erp_rgb = F.grid_sample(erp_hr.unsqueeze(0), erp_coord.unsqueeze(0).unsqueeze(0),
                                "bilinear", align_corners=False).squeeze(0).squeeze(1)
        fisheye_rgb[int(0.5 - i / 2), :, y_fisheye, x_fisheye] = erp_rgb

    ds_func = Image.Resampling.BICUBIC if downsample_type == 'bicubic' else Image.Resampling.BILINEAR
    fisheye_lr = torch.zeros([2, 3, fisheye_patch_size // scale,
                              fisheye_patch_size // scale])
    for i in [0, 1]:
        fisheye_lr[i] = transforms.ToTensor()(transforms.ToPILImage()(fisheye_rgb[i])
                                              .resize(fisheye_lr.shape[-2:], ds_func))

    # lookup table
    erp_lr = torch.zeros([3, h // scale, w // scale])
    h_fisheye_lr = h // scale // 2
    erp_coord_lr = make_coord([*erp_lr.shape[-2:]], flatten=False).reshape(2, h_fisheye_lr, w // scale, 2)

    for i in [-1, 1]:
        y_erp, x_erp = erp_coord_lr[int(0.5 - i / 2), :, :, 0].reshape(-1), \
                       erp_coord_lr[int(0.5 - i / 2), :, :, 1].reshape(-1)
        theta = x_erp * math.pi
        rho = 1 + y_erp * i
        rho /= fisheye_expand_scale
        _y_fisheye, _x_fisheye = rho * torch.cos(theta), rho * torch.sin(theta)
        _fisheye_coord = torch.stack([_y_fisheye, _x_fisheye], dim=-1)
        lr_rgb = F.grid_sample(fisheye_lr[int(0.5 + i / 2)].unsqueeze(0),
                               _fisheye_coord.unsqueeze(0).unsqueeze(0),
                               "bilinear",
                               align_corners=False).squeeze(0).squeeze(1)
        erp_lr[:, h_fisheye_lr * (1 - int(0.5 + i / 2)): h_fisheye_lr * (2 - int(0.5 + i / 2)), :] = \
            lr_rgb.reshape(3, h_fisheye_lr, w // scale)
    erp_lr = erp_lr.clamp(0, 1)
    return erp_lr


def erp_downsample_fisheye_xoz(erp_hr: torch.Tensor, scale, downsample_type='bicubic',
                           fisheye_expand_scale=1.25, fisheye_patch_scale=1.5, overlap_pixel=0, double_coord=True):
    """
    backward (implicit bicubic downsample process based on uniformed dual fisheye)
    :param fisheye_patch_scale: float, fisheye resolution [h * scale, h * scale]
    :param fisheye_expand_scale: expand 180 degree fisheye for edge consistency
    :param overlap_pixel: overlap sampling for boundary, max: width//scale//16
    :param erp_hr: (Tensor) [c, h, w]
    :param scale: downsample scale
    :param downsample_type: str, bicubic or bilinear
    :return:
        erp_lr (Tensor): lr Tensor [c, h, w] in [0, 1]
    """
    h, w = erp_hr.shape[-2:]
    erp_hr = erp_hr[:, :h // scale * scale, :w // scale * scale]

    fisheye_patch_size = round(h * fisheye_patch_scale * fisheye_expand_scale)

    # fisheye
    fisheye_rgb = torch.zeros([2, 3, fisheye_patch_size, fisheye_patch_size])
    fisheye_coord = make_coord([*fisheye_rgb.shape[-2:]], flatten=False, double=double_coord)
    fisheye_rho = torch.sqrt(fisheye_coord[:, :, 0] ** 2 + fisheye_coord[:, :, 1] ** 2)
    fisheye_rho *= fisheye_expand_scale
    fisheye_theta = torch.arctan2(fisheye_coord[:, :, 0], fisheye_coord[:, :, 1])
    x_fisheye, y_fisheye = np.where(fisheye_rho.numpy() <= fisheye_expand_scale)
    # expand z-dimension
    rho = fisheye_rho[y_fisheye, x_fisheye]
    theta = fisheye_theta[y_fisheye, x_fisheye]
    k = torch.tan(math.pi/2*(1-rho))
    x_erp = 1 - torch.arctan2(k, torch.cos(theta)) / math.pi
    y_erp = 2 * torch.arctan2(torch.sin(theta),torch.sqrt((torch.cos(theta)**2+k**2))) / math.pi


    for i in [-1, 1]:
        if i == -1:
            _erp_hr = torch.cat([erp_hr[:,:,w//2:], erp_hr[:,:,:w//2]], dim=-1)
        else:
            _erp_hr = erp_hr
        _erp_hr_padded = torch.zeros([3, h, w*2])
        _erp_hr_padded[:,:,:w//2] = _erp_hr[:,:,-w//2:]
        _erp_hr_padded[:, :, -w // 2:] = _erp_hr[:,:,:w//2]
        _erp_hr_padded[:, :, w // 2: w//2*3] = _erp_hr
        erp_coord = torch.stack([(x_erp - 1)*0.5, y_erp], dim=-1).float()
        erp_rgb = F.grid_sample(_erp_hr_padded.unsqueeze(0), erp_coord.unsqueeze(0).unsqueeze(0),
                                "bilinear", align_corners=False).squeeze(0).squeeze(1)
        fisheye_rgb[int(0.5 - i / 2), :, y_fisheye, x_fisheye] = erp_rgb

    ds_func = Image.Resampling.BICUBIC if downsample_type == 'bicubic' else Image.Resampling.BILINEAR
    fisheye_lr = torch.zeros([2, 3, fisheye_patch_size // scale,
                              fisheye_patch_size // scale])
    for i in [0, 1]:
        fisheye_lr[i] = transforms.ToTensor()(transforms.ToPILImage()(fisheye_rgb[i])
                                              .resize(fisheye_lr.shape[-2:], ds_func))

    # lookup table
    erp_lr = torch.zeros([3, h // scale, w // scale])
    _erp_lr = torch.zeros([2, 3, h // scale, w // scale // 2 + overlap_pixel * 2])
    erp_coord_lr = make_coord([*erp_lr.shape[-2:]], flatten=True, double=double_coord).reshape(h // scale, w // scale, 2)

    for i in [-1, 1]:
        erp_coord_lr_ext = torch.zeros([h // scale, w // scale // 2 + overlap_pixel * 2, 2],
                                       dtype= torch.float64 if double_coord else torch.float32)

        if i == 1:
            erp_coord_lr_ext[:, overlap_pixel:, :] = erp_coord_lr[:, :w // scale // 2 + overlap_pixel, :]
        else:
            erp_coord_lr_ext[:, :overlap_pixel + w // scale // 2, :] = erp_coord_lr[:, w // scale // 2 - overlap_pixel:, :]

        if overlap_pixel >0:
            if i == 1:
                erp_coord_lr_ext[:, :overlap_pixel, :] = erp_coord_lr[:, -overlap_pixel:, :]
            else:
                erp_coord_lr_ext[:, -overlap_pixel:, :] = erp_coord_lr[:, :overlap_pixel, :]

        y_erp, x_erp = erp_coord_lr_ext[:, :, 0].reshape(-1), \
                       erp_coord_lr_ext[:, :, 1].reshape(-1)
        x_s = torch.cos(y_erp * math.pi / 2) * torch.cos(x_erp * math.pi)
        y_s = torch.cos(y_erp * math.pi / 2) * torch.sin(x_erp * math.pi)
        z_s = torch.sin(y_erp * math.pi / 2)

        theta = torch.arctan2(z_s, x_s)
        rho = torch.arctan2(torch.sqrt(x_s**2+z_s**2), y_s) * 2 / math.pi - 2 * int(0.5 + i / 2)
        rho /= fisheye_expand_scale
        _y_fisheye, _x_fisheye = rho * torch.cos(theta), rho * torch.sin(theta)

        if i == 1:
            _x_fisheye *= -1

        _fisheye_coord = torch.stack([_y_fisheye, _x_fisheye], dim=-1).float()
        lr_rgb = F.grid_sample(fisheye_lr[int(0.5 + i / 2)].unsqueeze(0),
                               _fisheye_coord.unsqueeze(0).unsqueeze(0),
                               "bilinear",
                               align_corners=False).squeeze(0).squeeze(1)

        lr_rgb = lr_rgb.reshape(3, h // scale, w // scale // 2 + overlap_pixel * 2)
        if overlap_pixel > 0:
            lr_rgb[:, :, :overlap_pixel * 2] *= torch.linspace(0, 1, overlap_pixel * 2).reshape(1, 1, overlap_pixel * 2)
            lr_rgb[:, :,-overlap_pixel * 2:] *= torch.linspace(1, 0, overlap_pixel * 2).reshape(1, 1, overlap_pixel * 2)
        _erp_lr[int(0.5 + i / 2)] = lr_rgb

    erp_lr[:, :, :w // scale // 2 + overlap_pixel] += _erp_lr[1, :, :, overlap_pixel:]
    erp_lr[:, :, w // scale // 2 - overlap_pixel:] += _erp_lr[0, :, :, :overlap_pixel + w // scale // 2]
    if overlap_pixel > 0:
        erp_lr[:,:,-overlap_pixel:] += _erp_lr[1,:,:,:overlap_pixel]
        erp_lr[:,:,:overlap_pixel] += _erp_lr[0,:,:,-overlap_pixel:]

    erp_lr = erp_lr.clamp(0, 1)
    return erp_lr.flip(2)


if __name__ == '__main__':
    given_path = ''
    save_path = ''
    scale = 4
    erp_hr = transforms.ToTensor()(Image.open(given_path).convert('RGB'))
    erp_lr = erp_downsample_fisheye(erp_hr, scale)
    transforms.ToPILImage()(erp_lr).save(save_path)

