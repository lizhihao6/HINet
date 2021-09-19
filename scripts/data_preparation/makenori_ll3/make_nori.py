#!/usr/bin/env python3
import json
import os

import nori2 as nori
import numpy as np
import refile
from tqdm import tqdm

H = 720
W = 1280


def dir2nori(inp_dir, gt_dir, nori_file, json_dir):
    inputs = [x for x in refile.smart_glob(refile.smart_path_join(inp_dir, '*.png'))]
    gts = [x.replace(inp_dir, gt_dir) for x in inputs]
    nw = nori.remotewriteopen(nori_file)
    res = []

    def callback(nid, e):
        if e is not None:
            print(e)
            return
        res.append(nid)

    for i in tqdm(range(len(inputs))):
        # inp_img = cv2.imdecode(np.frombuffer(refile.smart_open(inp, 'rb').read(), np.uint8), cv2.IMREAD_UNCHANGED,)
        inp_img = np.frombuffer(refile.smart_open(inputs[i], 'rb').read(), dtype=np.uint8)
        gt_img = np.frombuffer(refile.smart_open(gts[i], 'rb').read(), dtype=np.uint8)
        inp_img_encode = inp_img.tobytes()
        gt_img_encode = gt_img.tobytes()
        nw.async_put(callback, inp_img_encode)
        nw.async_put(callback, gt_img_encode)
    nw.join()
    return res, inputs

def _convert(datasets):
    dataset = 'train'
    inp_dir, gt_dir, nori_file, json_file = datasets[dataset]
    # convert to nori
    res, img_paths = dir2nori(inp_dir, gt_dir, nori_file, json_file)

    # save json
    json_dir = os.path.dirname(json_file)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(json_file, 'w') as f:
        json.dump([res, img_paths], f)
        # json.dump(img_paths, f)

def convert_gopro():
    datasets = {
        'train': [
            's3://lzh-share/GoPro/train/input/',
            's3://lzh-share/GoPro/train/target/',
            's3://llcv-dataspace/GoPro/train.nori',
            './nori_json/train.json'
        ]
    }
    _convert(datasets)


def convert_stereo():
    datasets = {
        'train': [
            's3://lzh-share/stereo_blur_data/train/input/',
            '/data/stereo_blur_data/train/target/',
            's3://llcv-dataspace/stereo_blur_data/train.nori',
            '/data/stereo_blur_data/train/train_nori.json'
        ]
    }
    _convert(datasets)

if __name__ == "__main__":
    convert_stereo()

# vim: ts=4 sw=4 sts=4 expandtab
