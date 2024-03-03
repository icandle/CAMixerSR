from erp_downsample import erp_downsample_fisheye
from pers2erp import pers2erp
import glob as gb
import os
import os.path as osp
import cv2
import torch
from torchvision import transforms
from multiprocessing import Pool
from tqdm import tqdm

os.makedirs('/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI2/HR', exist_ok=True)
os.makedirs('/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI2/LR/X2', exist_ok=True)
os.makedirs('/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI2/LR/X4', exist_ok=True)
# os.makedirs('/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI2/LR/X8', exist_ok=True)
# os.makedirs('/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI/LR/X16', exist_ok=True)


def extract_odi_subimage(img_pth, save_pth, img_idx, ):
    lat_relative_aug = True
    crop_dict = {0: [[320, 704], [784, 1264]],
                 15: [[224, 640], [816, 1232]],
                 30: [[112, 576], [832, 1216]],
                 45: [[0, 512], [832, 1216]],
                 -15: [[384, 800], [816, 1232]],
                 -30: [[448, 912], [832, 1216]],
                 -45: [[512, 1024], [832, 1216]],
                 }

    img = cv2.imread(img_pth, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    repeat_num = int(w/256-2)
    sub_idx = 1
    for i in range(repeat_num):
        _img = img[:, i*256:(i+2)*256, :]
        for lat_sign in [-1, 0, 1]:
            if lat_sign == -1:
                _pers = _img[:512]
            elif lat_sign == 1:
                _pers = _img[-512:]
            else:
                _pers = _img[h//2-256:h//2+256]
            if lat_relative_aug:
                phi_list = [_phi + lat_sign * -30 for _phi in [-15, 0, 15]]
            else:
                phi_list = list(range(-45, 60, 15))
            for phi in phi_list:
                [[h0, h1], [w0, w1]] = crop_dict[phi]
                sub_name = f'{img_idx:04d}_s{sub_idx:03d}_hw~_{h0}_{w0}_~.png'

                erp_hr = pers2erp(_pers, phi)
                _erp_hr = torch.tensor(erp_hr, dtype=torch.float).permute(2, 0, 1)/ 255
                _erp_hr = _erp_hr[[2, 1, 0], :, :]
                for scale in [2, 4]:
                    _erp_lr = erp_downsample_fisheye(_erp_hr, scale)
                    _erp_lr = _erp_lr[:, h0//scale:h1//scale, w0//scale:w1//scale]
                    transforms.ToPILImage()(_erp_lr).save(osp.join(save_pth, 'LR/X%s' % scale, sub_name))

                _erp_hr = _erp_hr[:, h0:h1, w0:w1]
                transforms.ToPILImage()(_erp_hr).save(osp.join(save_pth, 'HR', sub_name))
                sub_idx += 1

def main():
    #mlx gpu launch --gpu=0 --cpu=25 --memory=256 --node=1  python3 odisr/utils/make_augmentation_dataset.py
    save_pth = '/mnt/bn/mmlab-wangyan-srdata/DF2K-ODI2'
    img_list = gb.glob('/mnt/bn/mmlab-wangyan-srdata/DIV2K/DIV2K_train_HR/*') + gb.glob('/mnt/bn/mmlab-wangyan-srdata/Flickr2K/Flickr2K_HR/*')
    n = 3
    k = 900
    p = 700
    start = n*k + p
    end = min((n+1)*k,len(img_list))
    img_list = img_list[start:end]
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(50)
    for i, path in enumerate(img_list):
        pool.apply_async(extract_odi_subimage, args=(path, save_pth, i+start, ), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


if __name__ == '__main__':
    main()