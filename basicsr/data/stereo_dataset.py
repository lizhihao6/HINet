# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

import json

import cv2
import numpy as np
import torch
from aiisp_tool.utils.oss_helper import OSSHelper
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, dvs_padding, dvs_paired_random_crop


class StereoImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(StereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        with open(opt['json_file'], 'r+') as f:
            self.json = json.load(f)
        self.get_keys = opt['get_keys']
        self.return_keys = opt['return_keys']
        assert len(self.get_keys) == len(self.return_keys), 'return keys length should be as same as get keys'
        self.helper, self.nf = None, None
        for k in self.json[0]:
            if ',' in self.json[0][k] and self.nf is None:
                import nori2 as nori
                from balls.imgproc import imdecode
                self.imdecode = imdecode
                self.nf = nori.Fetcher()

    def _load_img(self, im_path):
        if ',' in im_path:
            return self.imdecode(self.nf.get(im_path)).astype(np.float32) / 255.
        else:
            return cv2.imread(im_path).astype(np.float32) / 255.

    def _load_events(self, events_path):
        if 's3' in events_path:
            helper = OSSHelper()
            return helper.download(events_path, 'numpy').astype(np.float32)
        elif ',' in events_path:
            return np.concatenate([self.imdecode(self.nf.get(nid)) for nid in events_path.split('|')], axis=2).astype(
                np.float32) / 255. - 127. / 255.
        else:
            return np.load(events_path).astype(np.float32)

    def __getitem__(self, index):
        scale = self.opt['scale']
        meta = self.json[index]
        if self.opt['phase'] == 'train' and self.opt['random_swap_left_right']:
            for i in range(len(self.get_keys)):
                self.get_keys[i] = self.get_keys[i].replace('left', 'right') if 'left' in self.get_keys[i] else \
                    self.get_keys[i].replace('right', 'left')
        imgs = [self._load_img(meta[g]) if 'events' not in r else self._load_events(meta[g]) for g, r in
                zip(self.get_keys, self.return_keys)]

        # Load images and events. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            imgs = dvs_padding(imgs, gt_size)

            # random crop
            imgs = dvs_paired_random_crop(imgs, gt_size, scale, 'none')

            # flip, rotation
            imgs = augment(imgs, self.opt['use_flip'], self.opt['use_rot'])

            # to tensor
            imgs = img2tensor(imgs, bgr2rgb=True, float32=True)

            # chromatic transform
            color_imgs = [imgs[i] for i, v in enumerate(self.return_keys) if 'image' in v][::-1]
            for i, v in enumerate(self.return_keys):
                if 'image' in v:
                    imgs[i] = color_imgs.pop()

            # add noise
            inputs = [imgs[i] for i, v in enumerate(self.return_keys) if 'image' in v and 'gt' not in v]
            inputs = [im + np.sqrt(self.opt['noise_std']) * (torch.randn_like(im) - 0.5) for im in inputs][::-1]
            for i, v in enumerate(self.return_keys):
                if 'image' in v and 'gt' not in v:
                    imgs[i] = inputs.pop()

        # for test or val
        else:
            imgs = img2tensor(imgs, bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            imgs = [normalize(im, self.mean, self.std, inplace=True) for im in imgs]

        return_dict = {v: imgs[i] for i, v in enumerate(self.return_keys)}
        for g, r in zip(self.get_keys, self.return_keys):
            return_dict[r + '_path'] = meta[g]
        return return_dict

    def __len__(self):
        return len(self.json)
