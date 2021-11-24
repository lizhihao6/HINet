# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import json

import h5py
import numpy as np
import sparseconvnet as scn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data as data
from pytorch3d.ops import knn_gather, sample_farthest_points

class EventsVoxelSRDataset(data.Dataset):
    def __init__(self, opt):
        super(EventsVoxelSRDataset, self).__init__()
        self.opt = opt
        with open(opt['voxel_annotation'], 'r') as f:
            self.annotation = {}
            annotation = json.load(f)
            for k in annotation:
                self.annotation[k] = np.array([[v['start_id'], v['stop_id']] for v in annotation[k]], dtype=np.uint32)
        self._h5, self._map_h5, self.h5_path, self.map_h5_path = None, None, opt.get('h5'), opt.get('maplist')
        with open(opt['labels'], 'r') as f:
            self.pairs = json.load(f)
        self.sample_fn, self.sample_num = opt['sample_fn'], opt['sample_num']
        assert self.sample_fn in ['all', 'crop']
        # sample_fn: all -> return all windows; crop -> return subset windows;
        self.lq_to_frame, self.gt_to_frame = None, None
        self.return_events = opt.get('return_events', False)

    @property
    def h5(self):
        if self._h5 is None:  # lazy loading here!
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    @property
    def map_h5(self):
        if self._map_h5 is None:  # lazy loading here!
            self._map_h5 = h5py.File(self.map_h5_path, 'r')
        return self._map_h5

    def _set_events_to_frame(self, h, w):
        h, w = int(h), int(w)
        return scn.Sequential(
            scn.InputLayer(dimension=2, spatial_size=torch.LongTensor([h, w]), mode=3),  # sum
            scn.SparseToDense(2, 1)
        )

    def _get_size(self, lq_key, gt_key):
        lq_size, gt_size = json.loads(self.h5[lq_key + '_pos'].attrs['size']), json.loads(
            self.h5[gt_key + '_pos'].attrs['size'])
        lq_size, gt_size = np.array([[lq_size['width'], lq_size['height']]], dtype=np.float32), np.array(
            [[gt_size['width'], gt_size['height']]], dtype=np.float32)

        if self.lq_to_frame is None or self.gt_to_frame is None:
            self.lq_to_frame = self._set_events_to_frame(lq_size[0, 1], lq_size[0, 0])
            self.gt_to_frame = self._set_events_to_frame(gt_size[0, 1], gt_size[0, 0])
        return lq_size, gt_size

    def _get_split_events(self, k, size, start_id, stop_id) -> torch.FloatTensor:
        events = np.zeros([stop_id - start_id, 4], dtype=np.float32)
        events[:, :2] = self.h5[k + '_pos'][start_id:stop_id].astype(np.float32) / size
        events[:, 2] = self.h5[k + '_timestamps'][start_id:stop_id].astype(np.float32)
        events[:, 3] = self.h5[k + '_events'][start_id:stop_id]
        return torch.from_numpy(events)

    def _events_to_frame(self, k, start_ids: list, stop_ids: list, fn) -> torch.FloatTensor:
        xy = self.h5[k + '_pos'][start_ids[0]:stop_ids[-1]]
        p = self.h5[k + '_events'][start_ids[0]:stop_ids[-1]]
        yx = torch.from_numpy(xy[:, [1, 0]].astype(np.int64)).long()
        b = torch.zeros([yx.shape[0], 1]).long()
        for i in range(len(start_ids)):
            b[start_ids[i]:stop_ids[i]] = i
        yxb = torch.cat([yx, b], dim=-1)
        p = (torch.from_numpy(p).float().unsqueeze(1) - 0.5) / 0.5
        return fn([yxb, p])

    def _get_events(self, lq_key, gt_key):
        data_len = len(self.annotation[lq_key])
        if self.sample_fn == 'crop':
            voxel_start_id = np.random.randint(0, data_len - self.sample_num)
            voxel_stop_id = voxel_start_id + self.sample_num  # exclude
        else:
            voxel_start_id, voxel_stop_id = 0, data_len  # exclude

        data = {}
        # get lq
        lq_size, gt_size = self._get_size(lq_key, gt_key)
        start_id, stop_id = self.annotation[lq_key][voxel_start_id][0], self.annotation[lq_key][voxel_stop_id - 1][
            -1]  # exclude
        start_ids = self.annotation[lq_key][voxel_start_id:voxel_stop_id, 0].tolist()
        stop_ids = start_ids[1:] + [stop_id]  # exclude
        data['lq'] = self._events_to_frame(lq_key, start_ids, stop_ids, self.lq_to_frame).squeeze(1)
        if self.return_events:
            data['lq_events'] = self._get_split_events(lq_key, lq_size, start_id, stop_id)
            data['lq_frame_start_ids'] = torch.LongTensor(start_ids)
            data['lq_frame_stop_ids'] = torch.LongTensor(stop_ids)
            map_list = self.map_h5[lq_key][start_id:stop_id]
            # may cause map_list[lq_frame_stop_ids] != gt_frame_stop_ids
            data['map_list'] = torch.from_numpy(map_list.astype(np.int64)).long()

        start_id, stop_id = self.map_h5[lq_key][start_id], self.map_h5[lq_key][stop_id - 1] + 1  # exclude
        start_ids = self.map_h5[lq_key][start_ids].tolist()
        stop_ids = start_ids[1:] + [stop_id]
        data['gt'] = self._events_to_frame(gt_key, start_ids, stop_ids, self.gt_to_frame).squeeze(1)
        if self.return_events:
            data['gt_events'] = self._get_split_events(gt_key, gt_size, start_id, stop_id)
            data['gt_frame_start_ids'] = torch.LongTensor(start_ids)
            data['gt_frame_stop_ids'] = torch.LongTensor(stop_ids)
            data['map_list'] -= start_id
            assert data['map_list'][-1] == data['gt_events'].shape[0] - 1

        data['key'] = lq_key
        return data

    def __getitem__(self, index):
        pair = self.pairs[index]
        lq_key, gt_key = pair['lq'], pair['gt']
        return self._get_events(lq_key, gt_key)

    def __len__(self):
        return len(self.pairs)


# class EventsSRDataset(data.Dataset):
#     def __init__(self, opt):
#         super(EventsSRDataset, self).__init__()
#         self.opt = opt
#         self._h5, self._map_h5, self.h5_path, self.map_h5_path = None, None, opt['h5'], opt['maplist']
#         with open(opt['labels'], 'r') as f:
#             self.pairs = json.load(f)
#         self.sample_fn, self.sample_num = opt['sample_fn'], opt['sample_num']
#         assert self.sample_fn in ['all', 'crop']
#         # sample_fn: all -> return all windows; crop -> return subset windows;
#
#     @property
#     def h5(self):
#         if self._h5 is None:  # lazy loading here!
#             self._h5 = h5py.File(self.h5_path, 'r')
#         return self._h5
#
#     @property
#     def map_h5(self):
#         if self._map_h5 is None:  # lazy loading here!
#             self._map_h5 = h5py.File(self.map_h5_path, 'r')
#         return self._map_h5
#
#     def _get_events(self, lq_key, gt_key):
#         data_len = self.h5[lq_key + '_pos'].shape[0]
#         lq_size, gt_size = json.loads(self.h5[lq_key + '_pos'].attrs['size']), json.loads(
#             self.h5[gt_key + '_pos'].attrs['size'])
#         lq_size, gt_size = np.array([[lq_size['width'], lq_size['height']]], dtype=np.float32), np.array(
#             [[gt_size['width'], gt_size['height']]], dtype=np.float32)
#
#         if self.sample_fn == 'crop':
#             sample_num = self.sample_num
#             start_id = np.random.randint(0, data_len - self.sample_num)
#             stop_id = start_id + self.sample_num
#         else:
#             sample_num = data_len
#             start_id, stop_id = 0, data_len
#         _stop_id = stop_id if stop_id != data_len else -1
#         gt_start_id, gt_stop_id = self.map_h5[lq_key][start_id], self.map_h5[lq_key][_stop_id]
#         lq_events = np.zeros([sample_num, 4], dtype=np.float32)
#         gt_events = np.zeros([gt_stop_id - gt_start_id, 4], dtype=np.float32)
#
#         lq_events[:, :2] = self.h5[lq_key + '_pos'][start_id:stop_id].astype(np.float32) / lq_size
#         lq_events[:, 2] = self.h5[lq_key + '_timestamps'][start_id:stop_id].astype(np.float32)
#         # do not norm time if use all
#         if self.sample_fn != 'all':
#             lq_events[:, 2] = (lq_events[:, 2] - lq_events[0, 2]) / (lq_events[-1, 2] - lq_events[0, 2])
#         lq_events[:, 3] = self.h5[lq_key + '_events'][start_id:stop_id]
#
#         gt_events[:, :2] = self.h5[gt_key + '_pos'][gt_start_id:gt_stop_id].astype(np.float32) / gt_size
#         gt_events[:, 2] = self.h5[gt_key + '_timestamps'][gt_start_id:gt_stop_id].astype(np.float32)
#         # do not norm time if use all
#         if self.sample_fn != 'all':
#             if gt_events[-1, 2] == gt_events[0, 2]:
#                 return self._get_events(lq_key, gt_key)
#             gt_events[:, 2] = (gt_events[:, 2] - gt_events[0, 2]) / (gt_events[-1, 2] - gt_events[0, 2])
#         gt_events[:, 3] = self.h5[gt_key + '_events'][gt_start_id:gt_stop_id]
#
#         lq_events, gt_events = torch.from_numpy(lq_events), torch.from_numpy(gt_events)
#         data = {'lq': lq_events, 'gt': gt_events, 'key': lq_key}
#         return data
#
#     @staticmethod
#     def collate_fn(batch):
#         lq = torch.cat([item['lq'][None] for item in batch], dim=0)
#         gt = pad_sequence([item['gt'] for item in batch], batch_first=True, padding_value=-1000.)
#         gt_length = torch.LongTensor([item['gt'].shape[0] for item in batch])
#         key = [item['key'] for item in batch]
#         return {'lq': lq, 'gt': gt, 'gt_length': gt_length, 'key': key}
#
#     def __getitem__(self, index):
#         pair = self.pairs[index]
#         lq_key, gt_key = pair['lq'], pair['gt']
#         data = self._get_events(lq_key, gt_key)
#         return data
#
#     def __len__(self):
#         return len(self.pairs)


class EventsSRDataset(data.Dataset):
    def __init__(self, opt):
        super(EventsSRDataset, self).__init__()
        self.opt = opt
        self._h5, self._map_h5, self.h5_path, self.map_h5_path = None, None, opt['h5'], opt['maplist']
        with open(opt['labels'], 'r') as f:
            self.pairs = json.load(f)
        self.sample_fn, self.sample_num = opt['sample_fn'], opt['sample_num']
        assert self.sample_fn in ['all', 'crop']
        # sample_fn: all -> return all windows; crop -> return subset windows;

    @property
    def h5(self):
        if self._h5 is None:  # lazy loading here!
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    @property
    def map_h5(self):
        if self._map_h5 is None:  # lazy loading here!
            self._map_h5 = h5py.File(self.map_h5_path, 'r')
        return self._map_h5

    def _get_events(self, lq_key, gt_key):
        data_len = self.h5[lq_key + '_pos'].shape[0]
        lq_size, gt_size = json.loads(self.h5[lq_key + '_pos'].attrs['size']), json.loads(
            self.h5[gt_key + '_pos'].attrs['size'])
        lq_size, gt_size = np.array([[lq_size['width'], lq_size['height']]], dtype=np.float32), np.array(
            [[gt_size['width'], gt_size['height']]], dtype=np.float32)

        if self.sample_fn == 'crop':
            sample_num = self.sample_num
            start_id = np.random.randint(0, data_len - self.sample_num)
            stop_id = start_id + self.sample_num
        else:
            sample_num = data_len
            start_id, stop_id = 0, data_len
        _stop_id = stop_id if stop_id != data_len else -1
        gt_start_id, gt_stop_id = self.map_h5[lq_key][start_id], self.map_h5[lq_key][_stop_id]
        lq_events = np.zeros([sample_num, 4], dtype=np.float32)
        gt_events = np.zeros([gt_stop_id - gt_start_id, 4], dtype=np.float32)

        lq_events[:, :2] = self.h5[lq_key + '_pos'][start_id:stop_id].astype(np.float32) / lq_size
        lq_events[:, 2] = self.h5[lq_key + '_timestamps'][start_id:stop_id].astype(np.float32)
        # do not norm time if use all
        if self.sample_fn != 'all':
            lq_events[:, 2] = (lq_events[:, 2] - lq_events[0, 2]) / (lq_events[-1, 2] - lq_events[0, 2])
        lq_events[:, 3] = self.h5[lq_key + '_events'][start_id:stop_id]

        gt_events[:, :2] = self.h5[gt_key + '_pos'][gt_start_id:gt_stop_id].astype(np.float32) / gt_size
        gt_events[:, 2] = self.h5[gt_key + '_timestamps'][gt_start_id:gt_stop_id].astype(np.float32)
        # do not norm time if use all
        if self.sample_fn != 'all':
            if gt_events[-1, 2] == gt_events[0, 2]:
                return self._get_events(lq_key, gt_key)
            gt_events[:, 2] = (gt_events[:, 2] - gt_events[0, 2]) / (gt_events[-1, 2] - gt_events[0, 2])
        gt_events[:, 3] = self.h5[gt_key + '_events'][gt_start_id:gt_stop_id]

        lq_events, gt_events = torch.from_numpy(lq_events), torch.from_numpy(gt_events)
        n = lq_events.shape[0]*4
        if gt_events.shape[0] > n:
            gt_events = gt_events[:n]
        elif gt_events.shape[0] < n:
            while gt_events.shape[0] < n:
                sample_num = min(n - gt_events.shape[0], gt_events.shape[0])
                _, selected_indices = sample_farthest_points(gt_events[None, :, :3], K=sample_num)  # [1, sample_num]
                pad_events = knn_gather(gt_events[None], selected_indices[:, :, None])  # [1, sample_num, 1, 4]
                pad_events = pad_events[0, :, 0, :]
                gt_events = torch.cat([gt_events, pad_events], dim=0)
        data = {'lq': lq_events, 'gt': gt_events, 'key': lq_key}
        return data

    def __getitem__(self, index):
        pair = self.pairs[index]
        lq_key, gt_key = pair['lq'], pair['gt']
        data = self._get_events(lq_key, gt_key)
        return data

    def __len__(self):
        return len(self.pairs)
