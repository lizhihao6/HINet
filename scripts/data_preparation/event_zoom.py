import os
from pathlib import Path

import numpy as np
import json
from low_level import Preprocess
import random
import h5py

class EZPreprocess(Preprocess):
    def __init__(self):
        event_zoom_path = '/lzh/datasets/DVS/EventZoom/'
        super(EZPreprocess, self).__init__(dict(file_paths=[str(s) for s in Path(event_zoom_path).glob('*/*.txt')],
                                                h5_path=os.path.join(event_zoom_path, 'eventzoom.h5'),
                                                voxel_path=os.path.join(event_zoom_path, 'eventzoom.voxel.json'),
                                                map_path=os.path.join(event_zoom_path, 'eventzoom.map.h5')))

    def _get_size_from_key(self, k):
        if 'ev_hr' in k:
            w, h = 222, 124
        elif 'ev_lr' in k:
            w, h = 111, 62
        elif 'ev_llr' in k:
            w, h = 56, 31
        return h, w

    def _file_to_h5(self, file_path):
        data_lens = os.popen('more {} | wc -l'.format(file_path)).readlines()[0]
        data_lens = int(float(data_lens)) - 1
        key = '_'.join(file_path.split('/')[-2:])
        timestamps, pos, events = np.zeros([data_lens], dtype=np.uint32), np.zeros([data_lens, 2],
                                                                                   dtype=np.uint8), np.zeros(
            [data_lens], dtype=np.bool)
        with open(file_path, 'r+') as txt_file:
            counter = 0
            for l in txt_file.readlines():
                if len(l.split(' ')) != 4:
                    continue
                time, x, y, e = l.split('\n')[0].split(' ')
                timestamps[counter] = int(float(time))
                pos[counter, 0] = int(float(x)) - 1
                pos[counter, 1] = int(float(y)) - 1
                events[counter] = int(float(e))
                counter += 1
        return key, timestamps, pos, events

    def train_test_split(self, train_ratio=0.8):
        with h5py.File(self.voxel_path, 'r') as f:
            gt_keys = [k for k in f.keys() if 'hr' in k]
        train_gt_keys = random.sample(gt_keys, int(len(gt_keys) * train_ratio))
        test_gt_keys = [k for k in gt_keys if k not in train_gt_keys]
        train_2x_json_path = self.h5_path.replace(os.path.basename(self.h5_path), 'train_2x.json')
        test_2x_json_path = train_2x_json_path.replace('train', 'test')
        train_4x_json_path = train_2x_json_path.replace('2x', '4x')
        test_4x_json_path = test_2x_json_path.replace('2x', '4x')

        train_2x = [{'lq': key.replace('hr', 'lr_1'), 'gt':key} for key in train_gt_keys]
        train_2x += [{'lq': key.replace('hr', 'lr_2'), 'gt': key} for key in train_gt_keys]
        train_4x = [{'lq': key.replace('hr', 'llr_1'), 'gt':key} for key in train_gt_keys]
        train_4x += [{'lq': key.replace('hr', 'llr_2'), 'gt': key} for key in train_gt_keys]
        test_2x = [{'lq': key.replace('hr', 'lr_1'), 'gt':key} for key in test_gt_keys]
        test_2x += [{'lq': key.replace('hr', 'lr_2'), 'gt': key} for key in test_gt_keys]
        test_4x = [{'lq': key.replace('hr', 'llr_1'), 'gt':key} for key in test_gt_keys]
        test_4x += [{'lq': key.replace('hr', 'llr_2'), 'gt': key} for key in test_gt_keys]

        for pair, path in zip([train_2x, train_4x, test_2x, test_4x], [train_2x_json_path, train_4x_json_path, test_2x_json_path, test_4x_json_path]):
            with open(os.path.join(path), 'w+') as f:
                json.dump(pair, f)

if __name__ == '__main__':
    preprocessor = EZPreprocess()
    preprocessor.run(['h5_to_voxel'])
    # preprocessor.run(['h5_to_maplist'])
    # preprocessor.train_test_split()