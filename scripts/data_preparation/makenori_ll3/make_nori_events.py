#!/usr/bin/env python3
import refile
import nori2 as nori
import numpy as np
import os
import json
import pickle
import cv2
from tqdm import tqdm

H = 720
W = 1280


def dir2nori(inp_dir, nori_file, info_file):
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    inputs = [x for x in refile.smart_glob(refile.smart_path_join(inp_dir, '*.npy'))]
    nw = nori.remotewriteopen(nori_file)
    res = []
    info = {}

    def callback(nid, e):
        if e is not None:
            print(e)
            return
        res.append(nid)

    for i in tqdm(range(len(inputs))):
        img_name = os.path.basename(inputs[i])

        # load numpy array
        tmp_path = os.path.join(tmp_dir, img_name)
        cmd = 'aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp \'{}\' {}'.format(inputs[i], tmp_path)
        os.system(cmd)
        event_ori = np.load(tmp_path)
        event_array = event_ori.tobytes()
        os.remove(tmp_path)

        # load txt
        event_txt = refile.smart_open(inputs[i].replace('.npy', '.txt'), 'r').read()

        event_encode = pickle.dumps({'event_array': event_array, 'shape': event_ori.shape, 'event_txt': event_txt, 'name': img_name.replace('.npy', '.png')})
        nw.async_put(callback, event_encode)

    nw.join()

    for i, input_name in enumerate(inputs):
        info[os.path.basename(input_name).replace('.npy', '.png')] = res[i]
    pickle.dump(info, open(info_file, 'wb'))
    return res


def main():
    datasets = {
        'train': [
            's3://lzh-share/GoPro/train/events/',
            's3://llcv-dataspace/GoPro/train_events_v1.nori',
            './nori_json/train_events_v1.info'
        ]
    }
    dataset = 'train'
    inp_dir, nori_file, info_file = datasets[dataset]

    if not os.path.exists(os.path.dirname(info_file)):
        os.makedirs(os.path.dirname(info_file))
    # convert to nori
    res = dir2nori(inp_dir, nori_file, info_file)

    # save json
    # json_dir = os.path.dirname(json_file)
    # if not os.path.exists(json_dir):
    #     os.makedirs(json_dir)
    # with open(json_file, 'w') as f:
    #     json.dump([res, img_paths], f)
        # json.dump(img_paths, f)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
