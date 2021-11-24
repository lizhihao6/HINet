# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import json

import h5py
import numpy as np
import torch
from pytorch3d.ops import knn_gather, sample_farthest_points
from torch.utils import data as data

from basicsr.data.data_util import events_augmentation


class EventsClsReflectDataset(data.Dataset):
    def __init__(self, opt):
        super(EventsClsReflectDataset, self).__init__()
        self.opt = opt
        self._h5, self.h5_path = None, opt['h5']
        with open(opt['labels'], 'r') as f:
            self.labels = json.load(f)
            self.classes = self.labels.pop('classes')
            self.labels = [{'key': key, 'label_id': self.classes.index(self.labels[key])} for key in self.labels]
        # if 'test' in opt['labels']:
        #     self.labels = random.sample(self.labels, 10)

        self.sample_fn, self.sample_num = opt['sample_fn'], opt['sample_num']
        assert self.sample_fn in ['all', 'crop', 'farthest']
        assert self.sample_num > 0
        # sample_fn: all -> return all points; crop -> return subset point; farthest -> return farthest point
        self.augmentation = self.opt.get('augmentation', False)

    @property
    def h5(self):
        if self._h5 is None:  # lazy loading here!
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _get_events(self, key):
        data_len, size = self.h5[key + '_events'].shape[0], json.loads(self.h5[key + '_pos'].attrs['size'])
        size = np.array([[size['width'], size['height']]], dtype=np.float32)

        if self.sample_fn == 'crop':
            sample_size, target_size = min(data_len, self.sample_num), self.sample_num
            if data_len <= target_size:
                start_id, stop_id = 0, data_len
            else:
                start_id = np.random.randint(0, data_len - sample_size)
                stop_id = start_id + sample_size
        else:
            sample_size, target_size = data_len, max(data_len, self.sample_num)
            start_id, stop_id = 0, data_len
        events = np.zeros([sample_size, 4], dtype=np.float32)
        events[:, :2] = self.h5[key + '_pos'][start_id:stop_id].astype(np.float32) / size
        events[:, 2] = self.h5[key + '_timestamps'][start_id:stop_id].astype(np.float32)
        events[:, 3] = self.h5[key + '_events'][start_id:stop_id]

        # repeat padding
        while events.shape[0] % self.sample_num != 0:
            target_size = (events.shape[0] // self.sample_num +1) * self.sample_num
            sample_size = min(target_size-events.shape[0], events.shape[0])
            start_id = events.shape[0]-sample_size
            _events = events[start_id:].copy()
            _events[:, 2] = _events[:, 2] -_events[0, 2] + events[-1, 2]
            events = np.concatenate([events, _events], axis=0)

        # do not norm time if use all
        if self.sample_fn != 'all':
            events[:, 2] = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2])

        if self.augmentation:
            events = events_augmentation(events)
        events = torch.from_numpy(events)
        if self.sample_fn == 'farthest':
            events = events.to(0)
            _, selected_indices = sample_farthest_points(events[None, :, :3], K=self.sample_num)  # [1, sample_num]
            events = knn_gather(events[None], selected_indices[..., None])  # [1, sample_num, 1, 4]
            events = events[0, :, 0, :]
        return events

    def __getitem__(self, index):
        key = self.labels[index]['key']
        events = self._get_events(key)
        label_id = self.labels[index]['label_id']
        return {'events': events, 'gt': label_id, 'key': key}

    def __len__(self):
        return len(self.labels)
