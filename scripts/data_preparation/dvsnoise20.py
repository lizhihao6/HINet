import h5py
import numpy as np
from mat73 import loadmat

def mat73_to_h5(train_h5_path, test_h5_path, xyp_path, mat_path):
    mat = loadmat(mat_path)
    h = w = 25
    test_set = mat['setLabel'] == (mat['grpLabel'] - 1) * 3 + 2
    with h5py.File(train_h5_path, 'w') as f:
        f.create_dataset('timestamps', data=mat["X"][:, :, :, ~test_set])
        f.create_dataset('gt', data=mat["Y"][~test_set])
    with h5py.File(test_h5_path, 'w') as f:
        f.create_dataset('timestamps', data=mat["X"][:, :, :, test_set])
        f.create_dataset('gt', data=mat["Y"][test_set])
    xyp = np.zeros([h*w*8, 3], dtype=np.int8)
    counter = 0
    for x in range(h):
        for y in range(w):
            for p in [1 for _ in range(4)] + [-1 for _ in range(4)]:
                xyp[counter] = np.array([x, y, p])
                counter += 1
    np.save(xyp_path, xyp)


if __name__ == '__main__':
    train_h5_path = '/lzh/datasets/DVS/DVSNOISE20/train.h5'
    test_h5_path = '/lzh/datasets/DVS/DVSNOISE20/val.h5'
    xyp_path = '/lzh/datasets/DVS/DVSNOISE20/xyp.npy'
    mat_path = '/lzh/datasets/DVS/DVSNOISE20/all_labels.mat'
    mat73_to_h5(train_h5_path, test_h5_path, xyp_path, mat_path)
