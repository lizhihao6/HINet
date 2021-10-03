#!/usr/bin/env python3
import json
import multiprocessing as mp
import os
from multiprocessing import Pool

import cv2
import nori2 as nori
import numpy as np
import refile
from aiisp_tool.utils.oss_helper import OSSHelper
from balls.imgproc import imencode
from tqdm import tqdm


def _im_oss_to_nid(nw, helper, oss_png_path):
    if 's3' in oss_png_path:
        img = helper.download(oss_png_path, "bin")
        img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(oss_png_path)
    _, np4 = imencode('.np4', img)
    return nw.put(np4)


def _events_oss_to_nid(nw, helper, oss_events_path):
    if 's3' in oss_events_path:
        events = helper.download(oss_events_path, "numpy")
    else:
        events = np.load(oss_events_path)
    assert events.shape[2] == 16
    events = (events.astype(np.float32) + 127.).astype(np.uint8)
    nid = ""
    for i in range(0, events.shape[2] // 4):
        _, np4 = imencode('.np4', events[i * 4:i * 4 + 4])
        nid += (nw.put(np4) + '|')
    return nid[:-1]


def image2nori(input, gt, clean_events, noisy_events, nori_file):
    nw = nori.remotewriteopen(nori_file)
    helper = OSSHelper()
    meta = dict(
        left_base_name=os.path.basename(input),
        right_base_name=os.path.basename(input.replace('left', 'right')),
        left_blur_img_nid=_im_oss_to_nid(nw, helper, input),
        right_blur_img_nid=_im_oss_to_nid(nw, helper,
                                          input.replace('left', 'right')),
        left_sharp_img_nid=_im_oss_to_nid(nw, helper, gt),
        right_sharp_img_nid=_im_oss_to_nid(nw, helper,
                                           gt.replace('left', 'right')),
        left_clean_events_nid=_events_oss_to_nid(nw, helper, clean_events),
        right_clean_events_nid=_events_oss_to_nid(nw, helper, clean_events.replace('left', 'right')),
        left_noisy_events_nid=_events_oss_to_nid(nw, helper, noisy_events),
        right_noisy_events_nid=_events_oss_to_nid(nw, helper, noisy_events.replace('left', 'right')))
    return meta


def dir2nori(inp_dir, gt_dir, events_dir, nori_file, json_file):
    left_blur_paths = [
        x for x in refile.smart_glob(
            refile.smart_path_join(inp_dir, 'left*.png'))
    ]
    left_gt_paths = [x.replace(inp_dir, gt_dir) for x in left_blur_paths]
    if '_s' in left_blur_paths[0]:
        left_events_paths = [
            x.replace(inp_dir, events_dir).replace('_s', '.noisy_s').replace('png', 'npy') for x in left_blur_paths
        ]
    else:
        left_events_paths = [
            x.replace(inp_dir, events_dir).replace('png', 'noisy.npy') for x in left_blur_paths
        ]
    res = []
    metas = []

    pbar = tqdm(total=len(left_blur_paths), unit='image', desc='To nori')
    pool = Pool(mp.cpu_count())
    for i in range(len(left_blur_paths)):
        _res = pool.apply_async(
            image2nori,
            args=(
                left_blur_paths[i], left_gt_paths[i], left_events_paths[i].replace('noisy', 'clean'),
                left_events_paths[i],
                nori_file),
            callback=lambda arg: pbar.update(1))
        res.append(_res)
    for _res in res:
        metas.append(_res.get())
    pbar.close()

    json_dir = os.path.dirname(json_file)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(json_file, 'w') as f:
        json.dump(metas, f)


def convert_stereo():
    dir2nori('s3://lzh-share/stereo_blur_data/train/blur_crops',
             's3://lzh-share/stereo_blur_data/train/sharp_crops',
             's3://lzh-share/stereo_blur_data/train/events_crops',
             's3://llcv-dataspace/stereo_blur_data/train_v3.nori',
             './datasets/stereo_blur_data/train_v3.nori.json')

    dir2nori('s3://lzh-share/stereo_blur_data/test/input',
             '/data/stereo_blur_data/test/target',
             's3://lzh-share/stereo_blur_data/test/events',
             's3://llcv-dataspace/stereo_blur_data/test_v3.nori',
             './datasets/stereo_blur_data/test_3.nori.json')


if __name__ == "__main__":
    convert_stereo()

# vim: ts=4 sw=4 sts=4 expandtab
