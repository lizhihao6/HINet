# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
from multiprocessing import Pool
from os import path as osp

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from aiisp_tool.utils.oss_helper import OSSHelper

from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_gopro
from .dvs_genertor import stereo_generate_pairs, DVS_Genertor
from .makenori_ll3.make_nori import convert_stereo

def main():
    # gopro_pairs = stereo_generate_pairs()
    # dvs_genertor = DVS_Genertor(gopro_pairs)
    # dvs_genertor.run(["sharps_to_blur", "sharps_to_avi", "avi_to_events", "events_to_voxel"])

    opt = {}
    opt['n_thread'] = int(mp.cpu_count())
    opt['compression_level'] = 3
    opt['suffix'] = "png"

    opt['input_folder'] = 's3://lzh-share/stereo_blur_data/train/input'
    opt['save_folder'] = 's3://lzh-share/stereo_blur_data/train/blur_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = '/data/stereo_blur_data/train/target'
    opt['save_folder'] = 's3://data/stereo_blur_data/train/sharp_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = 's3://data/stereo_blur_data/train/events'
    opt['save_folder'] = 's3://data/stereo_blur_data/train/events_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    opt['suffix'] = "npy"
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
    # else:
    # print(f'Folder {save_folder} already exists. Exit.')
    # sys.exit(1)

    img_list = list(scandir(input_folder, suffix=opt['suffix'], full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
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
    helper = OSSHelper()
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2',
                                '').replace('x3',
                                            '').replace('x4',
                                                        '').replace('x8', '')

    if "npy" not in path:
        if "s3" in path:
            img = helper.download(path, "bin")
            img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        if "s3" in path:
            img = helper.download(path, "numpy")
        else:
            img = np.load(path)
        

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

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
            save_path = osp.join(opt['save_folder'],
                                f'{img_name}_s{index:03d}{extension}')
            if "npy" not in path:
                if "s3" not in path:
                    cv2.imwrite(save_path, cropped_img,
                        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
                else:
                    cropped_img = cv2.imencode(".png", cropped_img)[1].tostring()
                    helper.upload(cropped_img, save_path, "bin")
            else:
                if "s3" not in path:
                    np.save(save_path, cropped_img)
                else:
                    helper.upload(cropped_img, save_path, "numpy")
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
