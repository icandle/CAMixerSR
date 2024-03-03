from os import path as osp
import os
import shutil
import glob as gb
from erp_downsample import erp_downsample_fisheye
from torchvision import transforms
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

'''
======================================
clean_datasets
======================================
'''

src = '/mnt/bn/mmlab-wangyan-srdata/lau_dataset'
dst = '/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean'
os.makedirs(dst, exist_ok=True)
train_dict = {'016': 'part of ERP', '026': 'virtual scenario', '027': 'virtual scenario', '028': 'virtual scenario',
              '029': 'virtual scenario', '030': 'virtual scenario', '031': 'virtual scenario',
              '032': 'virtual scenario',
              '033': 'virtual scenario', '034': 'virtual scenario', '035': 'virtual scenario',
              '040': 'virtual scenario',
              '051': 'virtual scenario', '053': 'virtual scenario', '054': 'virtual scenario',
              '055': 'virtual scenario',
              '057': 'virtual scenario', '060': 'virtual scenario', '124': 'mistakes in transform',
              '131': 'virtual scenario',
              '169': 'virtual scenario', '320': 'mistakes in transform', '321': 'mistakes in transform',
              '322': 'mistakes in transform', '329': 'mistakes in transform', '354': 'mistakes in transform',
              '360': 'mistakes in transform', '361': 'mistakes in transform', '362': 'mistakes in transform',
              '391': 'mistakes in transform', '409': 'mistakes in transform', '453': 'mistakes in transform',
              '458': 'mistakes in transform', '506': 'virtual scenario', '522': 'mistakes in transform', '541':
                  'part of ERP', '554': 'extremely poor quality', '559': 'extremely poor quality',
              '634': 'extremely poor quality',
              '644': 'extremely poor quality', '676': 'mistakes in transform', '813': 'mistakes in transform',
              '853': 'virtual scenario', '915': 'virtual scenario', '1049': 'mistakes in transform',
              '1055': 'mistakes in transform', '1072': 'virtual scenario', '1177': 'virtual scenario',
              '1184': 'mistakes in transform', '1196': 'extremely poor quality'}
validation_dict = {}
suntest_dict = {}
test_dict = {'014': 'virtual senario', '023': 'mistakes in transform', '045': 'not ERP'}

_dict = {'odisr/training': train_dict, 'odisr/testing': test_dict,
         'odisr/validation': validation_dict, 'sun_test': suntest_dict}

for split_type, rm_dict in _dict.items():
    img_paths = gb.glob(src + '/%s/HR/*' % split_type)
    for i, img_path in enumerate(img_paths):
        img_idx = osp.splitext(osp.split(img_path)[1])[0]
        relative_path = img_path.split('lau_dataset/')[-1]
        if img_idx in rm_dict.keys():
            print('rm %s: %s' % (img_idx, rm_dict[img_idx]))
            continue
        try:
            shutil.copy(img_path, osp.join(dst, relative_path))
        except FileNotFoundError:
            os.makedirs(osp.split(osp.join(dst, relative_path))[0], exist_ok=True)
            shutil.copy(img_path, osp.join(dst, relative_path))
        if 'jpg' in relative_path:
            os.rename(osp.join(dst, relative_path), osp.join(dst, relative_path[:-3] + 'png'))
        print('[%s][%s/%s]' % (split_type, i, len(img_paths)))


'''
======================================
downsampling
======================================
'''

HR_paths = ['/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/odisr/training/HR',
            '/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/odisr/validation/HR',
            '/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/odisr/testing/HR',
            '/mnt/bn/mmlab-wangyan-srdata/lau_dataset_clean/sun_test/HR',
            ]


def worker(img_path, scale, downsample_type, ):
    img_dst = osp.join(_root, osp.splitext(osp.split(img_path)[1])[0] + '.png')
    erp_hr = transforms.ToTensor()(transforms.Resize((1024, 2048))(Image.open(img_path).convert('RGB')))
    erp_lr = erp_downsample_fisheye(erp_hr, scale, downsample_type)
    transforms.ToPILImage()(erp_lr).save(img_dst)


for HR_path in HR_paths:
    imgs_path = gb.glob(HR_path+'/*')
    imgs_path = sorted(imgs_path)
    lr_ext = 'fisheye'
    downsample_type = 'bicubic'
    for scale in [2, 4, 8, 16]:
        _root = osp.join(osp.split(osp.split(imgs_path[0])[0])[0], 'LR_%s/X%s' % (lr_ext, scale))
        os.makedirs(_root, exist_ok=True)
        print(_root)
        pbar = tqdm(total=len(imgs_path), unit='image', desc='Downsampling')
        pool = Pool(20)
        for img_path in imgs_path:
            pool.apply_async(worker, args=(img_path, scale, downsample_type,), callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()
        pbar.close()
