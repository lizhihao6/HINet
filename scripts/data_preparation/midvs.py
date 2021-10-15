# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import json

def main(json_path='/data/MiDVS/test.json'):
    events = [str(s) for s in Path('/data/MiDVS/').glob('*/events_remap.npy')]
    cis = [p.replace('events_remap.npy', 'cis_remap.png') for p in events]
    metas = []
    for e, c in zip(events, cis):
        metas.append(
            dict(
                left_base_name=os.path.basename(e.split('/')[-2]),
                right_base_name=os.path.basename(e.split('/')[-2]),
                left_blur_img_path = c,
                left_sharp_img_path = c.replace('cis', 'baselines'),
                right_noisy_events_path = e)
        )
    with open(json_path, 'w+') as f:
        json.dump(metas, f)

if __name__ == '__main__':
    main()
