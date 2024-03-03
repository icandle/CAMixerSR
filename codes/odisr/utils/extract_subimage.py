import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from basicsr.utils import scandir


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.
    It is used for DIV2K dataset.
    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.
    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR
            DIV2K_train_LR_bicubic/X2
            DIV2K_train_LR_bicubic/X3
            DIV2K_train_LR_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages.
        /mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/odisr/training/LR_fisheye
        Remember to modify opt configurations according to your settings.
    """
    root_path = '/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/odisr/training/'
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    opt['input_folder'] = root_path + 'HR'
    opt['save_folder'] = root_path + 'HR_sub'
    opt['wh'] = (2048, 1024)
    opt['scale'] = 1
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = root_path + 'LR_fisheye/X2'
    opt['save_folder'] = root_path + 'LR_fisheye/X2_sub'
    opt['scale'] = 2
    opt['wh'] = (1024, 512)
    opt['crop_size'] = 256
    opt['step'] = 128
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = root_path + 'LR_fisheye/X4'
    opt['save_folder'] = root_path + 'LR_fisheye/X4_sub'
    opt['scale'] = 4
    opt['wh'] = (512, 256)
    opt['crop_size'] = 128
    opt['step'] = 64
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = root_path + 'LR_fisheye/X8'
    opt['save_folder'] = root_path + 'LR_fisheye/X8_sub'
    opt['scale'] = 8
    opt['wh'] = (256, 128)
    opt['crop_size'] = 64
    opt['step'] = 32
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = root_path + 'LR_fisheye/X16'
    opt['save_folder'] = root_path + 'LR_fisheye/X16_sub'
    opt['scale'] = 16
    opt['wh'] = (128, 64)
    opt['crop_size'] = 32
    opt['step'] = 16
    opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.
    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.
    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.
    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    input_shape = opt['wh']
    scale = opt['scale']

    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if input_shape is not None:
        img = cv2.resize(img, input_shape, cv2.INTER_LINEAR)
    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}_hw~_{int(x*scale)}_{int(y*scale)}_~{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()