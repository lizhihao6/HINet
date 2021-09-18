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
# from basicsr.utils.create_lmdb import create_lmdb_for_midvs


def main():
    # create_lmdb_for_midvs()
    for _p in tqdm(os.listdir("./datasets/MiDVS/")):
        p = os.path.join("./datasets/MiDVS", _p)
        if len([i for i in os.listdir(p) if i.endswith("txt")]) == 0:
            continue
        events = np.zeros([720, 960])
        for t in [str(s) for s in Path(p).glob("txt")]:
            with open(t, "r+") as f:
                for l in f.readlines():
                    y, x, p = l.split("\n")[0].split(" ")[1:]
                    y, x, p = int(float(y)), int(float(x)), int(float(p))
                    if p == 0:
                        events[y, x] -= 0.2
                    else:
                        events[y, x] += 0.2
        np.save("./datasets/MiDVS/events/{}.npy".format(p.split("/")[-2]), events)


if __name__ == '__main__':
    main()
