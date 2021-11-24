import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

num_cores = int(cpu_count()) //2


class Preprocess(ABC):
    def __init__(self, paths):
        '''
        Args:
            h5_path:
        '''
        for p in paths:
            self.__setattr__(p, paths[p])
        self.PIPELINE = {'files_to_h5': self.files_to_h5, 'h5_to_voxel': self.h5_to_voxel,
                         'h5_to_maplist': self.h5_to_maplist}

    def run(self, pipeline):
        for p in pipeline:
            assert p in self.PIPELINE.keys()
            self.PIPELINE[p]()

    @abstractmethod
    def _file_to_h5(self, file_path):
        # return key, timestamps, pos, events
        raise NotImplementedError

    @staticmethod
    def _get_size_from_key(k):
        # return h, w
        raise NotImplementedError

    def files_to_h5(self):
        f = h5py.File(self.h5_path, 'w')
        for path in tqdm(self.file_paths):
            key, timestamps, pos, events = self._file_to_h5(path)
            f.create_dataset('{}_timestamps'.format(key), data=timestamps)
            dset = f.create_dataset('{}_pos'.format(key), data=pos)
            h, w = self._get_size_from_key(key)
            dset.attrs['size'] = json.dumps(dict(height=h, width=w))
            f.create_dataset('{}_events'.format(key), data=events)
        f.close()

    def h5_to_voxel(self, time_resolution=10000):
        '''
        Args:
            h5_path: event zoom h5 path
            voxel_path: saved voxel path
            time_resolution: time used per channel (us)

        Returns:
            None
        '''
        read_file = h5py.File(self.h5_path, 'r')
        keys = [k for k in read_file.keys() if 'timestamps' in k]
        read_file.close()
        annotations = {}
        for k in tqdm(keys):
            annotations[k.replace('_timestamps', '')] = self._h5_to_voxel(k, time_resolution=time_resolution)
        with open(self.voxel_path, 'w+') as f:
            json.dump(annotations, f)

    def h5_to_maplist(self):
        '''
        Args:
            h5_path: event zoom h5 path
            voxel_path: saved voxel path
            time_resolution: time used per channel (us)

        Returns:
            None
        '''
        read_file = h5py.File(self.h5_path, 'r')
        out_file = h5py.File(self.map_path, 'w')
        keys = [k for k in read_file.keys() if 'timestamps' in k and 'hr' not in k]
        read_file.close()
        pbar = tqdm(total=len(keys), unit='numpy', desc='Maplist')
        for k in keys:
            lq_key, gt_key = k, 'ev_hr_' + '_'.join(k.split('_')[3:])
            map_list = self._h5_to_maplist(lq_key, gt_key)
            out_file.create_dataset(k.replace('_timestamps', ''), data=map_list)
            pbar.update(1)
        out_file.close()

    def _h5_to_voxel(self, k, time_resolution=10000):
        read_file = h5py.File(self.h5_path, 'r')
        total_frames = (read_file[k][-1] - read_file[k][0]) // time_resolution
        annotations = []
        pointer = 0
        cache = read_file[k].shape[0] // total_frames * 10
        for t in range(read_file[k][0]+time_resolution, read_file[k][-1], time_resolution):
            diff = np.abs(read_file[k][pointer:pointer+cache]-float(t))
            _pointer = np.argmin(diff)+pointer
            if read_file[k][_pointer] > t:
                _pointer -= 1
            if (pointer != 0 and _pointer == pointer) or _pointer < 0:  # no data in this frame
                print(pointer, _pointer, t, read_file[k][_pointer], cache)
                continue
            assert pointer < _pointer < pointer+cache or pointer == 0 or pointer == read_file[k].shape[0], 'cache small'
            annotations.append({'start_id': int(pointer), 'stop_id': int(_pointer)})
            pointer = _pointer
        assert len(annotations)>0, 'no frame data in {}'.format(k)
        return annotations

    def _h5_to_maplist(self, lq_key, gt_key):
        step, ratio = 64, 40
        read_file = h5py.File(self.h5_path, 'r')
        map_list = np.zeros(read_file[lq_key].shape[0], dtype=np.uint32)
        lq_dataset = read_file[lq_key]
        gt_dataset = read_file[gt_key]
        gt_start_id = 0
        for i in range(0, lq_dataset.shape[0], step):
            lq_start_id, lq_stop_id = i, min(i+step, lq_dataset.shape[0])
            gt_stop_id = min(gt_start_id+step*ratio, gt_dataset.shape[0])
            lq_ts = lq_dataset[lq_start_id:lq_stop_id].astype(np.int64)[:, None]
            gt_ts = gt_dataset[gt_start_id:gt_stop_id].astype(np.int64)[None, :]
            index = gt_start_id + np.argmin(np.abs(lq_ts-gt_ts), axis=1)
            map_list[lq_start_id:lq_stop_id] = index
            gt_start_id = int(index.max())
        read_file.close()
        return map_list
