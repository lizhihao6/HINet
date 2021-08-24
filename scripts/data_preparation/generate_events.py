import os
import cv2
from tqdm import trange
import numpy as np
import multiprocessing as mp
GOPRO_ORI_PATH = "/shared/GOPRO_Large/"
GOPRO_PATH = "/shared/GOPRO/"
V2E_PATH = "/lzh/Project/DVS/v2e"
SLOMO_CHECKPOINT="{}/input/SuperSloMo39.ckpt".format(V2E_PATH)
POS_THRES, NEG_THRES = .15, .15
FPS = 120
SIZE = (1280, 720)


COMMAND = "python {}/v2e.py " \
          "-i %(input)s " \
          "-o /tmp/output/$(date) " \
          "--avi_frame_rate=120 --overwrite --auto_timestamp_resolution " \
          "--output_height 720 --output_width 1280  --dvs_params clean --pos_thres={} --neg_thres={} " \
          "--dvs_emulator_seed=0 --slomo_model={} --no_preview " \
          "--dvs_text=%(output)s > /dev/null 2>&1".format(V2E_PATH, POS_THRES, NEG_THRES, SLOMO_CHECKPOINT)

def ims_to_avi(im_paths, save_path):
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, SIZE)
    # out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'HDYV'), FPS, SIZE)
    for p in im_paths:
        out.write(cv2.imread(p))
    out.release()

def avi_to_events(avi_path, save_path, idx):
    cmd = "CUDA_VISIBLE_DEVICES={} ".format(idx%2+2) + COMMAND%{'input':avi_path, 'output':save_path}
    os.system(cmd)
    events = np.zeros((SIZE[1], SIZE[0])).astype(np.float32)
    with open(save_path, "r+") as f:
        lines = [i for i in f.readlines() if not i.startswith("#")]
    for l in lines:
        _l = [int(float(i)) for i in l.split("\n")[0].split(" ")[1:]]
        x, y, p = _l[0], _l[1], _l[2]
        if p>0:
            events[y, x] += POS_THRES
        else:
            events[y, x] -= NEG_THRES
    np.save(save_path.replace(".txt", ".npy"), events)

def convert(prefix, name, show_bar, idx):
    assert prefix in ["train", "test"]
    ori_ids = [int(float(str(s)[:-len(".png")])) for s in os.listdir(os.path.join(GOPRO_ORI_PATH, prefix, name))]
    tar_ids = [int(float(str(s)[len(name+"-"):-len(".png")])) for s in os.listdir(os.path.join(GOPRO_PATH, prefix, "target")) if name in s]
    save_dir = os.path.join(GOPRO_PATH, prefix, "events")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ori_ids, tar_ids = sorted(ori_ids), sorted(tar_ids)
    steps = len(ori_ids) // len(tar_ids)
    assert len(ori_ids)==len(tar_ids)*steps, (prefix, name, len(ori_ids), len(tar_ids))
    if show_bar:
        iter = trange(len(tar_ids))
    else:
        iter = range(len(tar_ids))
    for i in iter:
        save_id = tar_ids[i]
        im_paths = [os.path.join(GOPRO_ORI_PATH, prefix, name, "%06d.png"%ori_id) for ori_id in ori_ids[i*steps:i*steps+steps]]
        avi_path = os.path.join(save_dir, "{}-".format(name)+"%06d.avi"%save_id)
        events_path = avi_path.replace(".avi", ".txt")
        ims_to_avi(im_paths, avi_path)
        avi_to_events(avi_path, events_path, idx)

if __name__ == '__main__':
    # num_cores = int(mp.cpu_count())
    num_cores = 4
    print("num cores: {}".format(num_cores))
    pool = mp.Pool(num_cores)
    # params = [["train", name, True] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "train"))]
    # params += [["test", name, True] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "test"))]
    params = [["test", name, True] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "test"))]
    # params[0][2] = True
    results = [pool.apply_async(convert, args=(t[0], t[1], t[2], i)) for i, t in enumerate(params)]
    results = [p.get() for p in results]
