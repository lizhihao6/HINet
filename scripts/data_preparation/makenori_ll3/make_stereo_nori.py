#!/usr/bin/env python3
import json
import os

import nori2 as nori
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import refile
from tqdm import tqdm
from balls.imgproc import imencode
from aiisp_tool.utils.oss_helper import OSSHelper
import cv2


def _oss_to_nid(nw, helper, oss_png_path):
    img = helper.download(oss_png_path, "bin")
    img = cv2.imdecode(
        np.fromstring(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    _, np4 = imencode('.np4', img)
    return nw.put(np4)


def image2nori(input, gt, events, nw):
    helper = OSSHelper()
    meta = dict(
        left_blur_img_nid=_oss_to_nid(nw, helper, input),
        right_blur_img_nid=_oss_to_nid(nw, helper,
                                       input.replace('left', 'right')),
        left_sharp_img_nid=_oss_to_nid(nw, helper, gt),
        right_sharp_img_nid=_oss_to_nid(nw, helper,
                                        gt.replace('left', 'right')),
        left_events_path=events,
        right_events_path=events.replace('left', 'right'))
    return meta


def dir2nori(inp_dir, gt_dir, events_dir, nori_file, json_file):
    left_blur_paths = [
        x for x in refile.smart_glob(
            refile.smart_path_join(inp_dir, 'left*.png'))
    ]
    left_gt_paths = [x.replace(inp_dir, gt_dir) for x in left_blur_paths]
    left_events_paths = [
        x.replace(inp_dir, events_dir) for x in left_blur_paths
    ]
    nw = nori.remotewriteopen(nori_file)
    res = []
    metas = []

    pbar = tqdm(total=len(left_blur_paths), unit='image', desc='To nori')
    pool = Pool(mp.cpu_count())
    for i in range(len(left_blur_paths)):
        _res = pool.apply_async(
            image2nori,
            args=(left_blur_paths[i], left_gt_paths[i], left_events_paths[i],
                  nw),
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
             's3://llcv-dataspace/stereo_blur_data/train.nori',
             './datasets/stereo_blur_data/train.nori.json')


if __name__ == "__main__":
    convert_stereo()

# vim: ts=4 sw=4 sts=4 expandtab
