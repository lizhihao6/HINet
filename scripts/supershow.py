#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
import refile
from balls.supershow2 import Submitter
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, type=str)
parser.add_argument("-t", "--tag", required=True, type=str)
args = parser.parse_args()

if __name__ == '__main__':
    img_gts = [x for x in refile.smart_glob(refile.smart_path_join(args.path, '*_gt.png'))]
    img_gts = sorted(img_gts)
    s = Submitter('single_deblur_ret')
    for i in trange(len(img_gts)):
        img_gt_path = img_gts[i]
        img_path = img_gt_path.replace('_gt', '')
        input_path = os.path.join(
            "./datasets/MiDVS/{}/\'and_Blur(original).png\'".format(os.path.basename(img_path)[:-4]))
        img_gt = cv2.imdecode(np.frombuffer(refile.smart_open(img_gt_path, 'rb').read(), dtype=np.uint8),
                              cv2.IMREAD_UNCHANGED, ).reshape(3000, 4000, 3)
        img = cv2.imdecode(np.frombuffer(refile.smart_open(img_path, 'rb').read(), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED, ).reshape(3000, 4000, 3)

        s.submit('{}'.format(args.tag),
                 {
                     "baseline": img_gt,
                     "our_ret": img
                 },
                 post_key="{}_{}".format(i, os.path.basename(img_path)))
