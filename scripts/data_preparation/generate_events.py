import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
from imageio import imread, imwrite
from tqdm import trange

GOPRO_ORI_PATH = "./datasets/GOPRO_Large/"
GOPRO_PATH = "./datasets/GoPro/"
V2E_PATH = "./.v2e"
SLOMO_CHECKPOINT = "{}/.pretrain/SuperSloMo39.ckpt".format(V2E_PATH)
POS_THRES, NEG_THRES = .2, .2  # use v2e --dvs_params clean will overwrite the --pos_thres and --neg_thres to .2
FPS = 120
SIZE = (1280, 720)
GPUS = 8

COMMAND = "python3 {}/v2e.py " \
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


def avi_to_events(avi_path, save_path):
    save_path = os.path.join(os.getcwd(), save_path)
    cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPUS) + COMMAND % {'input': avi_path, 'output': save_path}
    os.system(cmd)
    events = np.zeros((SIZE[1], SIZE[0])).astype(np.float32)
    with open(save_path, "r+") as f:
        lines = [i for i in f.readlines() if not i.startswith("#")]
    for l in lines:
        _l = [int(float(i)) for i in l.split("\n")[0].split(" ")[1:]]
        x, y, p = _l[0], _l[1], _l[2]
        if p > 0:
            events[y, x] += POS_THRES
        else:
            events[y, x] -= NEG_THRES
    np.save(save_path.replace(".txt", ".npy"), events)


def events_to_im(events_path, im_num, blurred_im_path, save_path):
    assert im_num % 2 != 0, "im_num should be odd"
    blurred_im = imread(blurred_im_path).astype(np.float32) / 255.
    linear_blurred_im = np.power(blurred_im, 2.2)
    y_blurred_im = linear_blurred_im*np.array([0.183, 0.614, 0.0624], dtype=np.float32).sum(2)*255.+16.
    lin_log_blurred_im = np.where(y_blurred_im<20, y_blurred_im*np.log(20)/20., np.log(y_blurred_im))
    lin_log_blurred_im = np.round(lin_log_blurred_im*1e8)/1e8

    events = [np.zeros_like(lin_log_blurred_im) for _ in range(im_num - 1)]
    diff, half_time = 1. / float(FPS), float(im_num // 2) / float(FPS)
    with open(events_path, "r+") as f:
        lines = [i for i in f.readlines() if not i.startswith("#")]
    for l in lines:
        _l = [float(i) for i in l.split("\n")[0].split(" ")]
        t, x, y, p = _l[0], int(_l[1]), int(_l[2]), int(_l[3])
        if t > diff * (im_num - 1):
            continue
        start_id = 0 if t < half_time else np.floor(t / diff)
        stop_id = np.ceil(t / diff) if t < half_time else im_num - 1
        for e in events[start_id:stop_id]:
            _p = -p if t < half_time else p
            if _p > 0:
                e[y, x] += POS_THRES
            else:
                e[y, x] -= NEG_THRES

    events = np.concatenate([e[np.newaxis, ...] for e in events], axis=0).sum(0) / (im_num - 1)
    assert events.shape == lin_log_blurred_im.shape, "events shape is consistent with blurred image shape"
    lin_log_sharp_im = lin_log_blurred_im / events
    y_sharp_im = np.where(lin_log_sharp_im<np.log(20.), lin_log_sharp_im*20/np.log(20.), np.exp(lin_log_sharp_im))
    y_sharp_im = np.clip(y_sharp_im, 16, 235)
    gray_sharp_im = np.exp(y_sharp_im/255., 1/2.2)*255.
    imwrite(save_path, gray_sharp_im.astype(np.uint8))


def convert(prefix, name, show_bar):
    assert prefix in ["train", "test"]
    ori_ids = [int(float(str(s)[:-len(".png")])) for s in os.listdir(os.path.join(GOPRO_ORI_PATH, prefix, name))]
    tar_ids = [int(float(str(s)[len(name + "-"):-len(".png")])) for s in
               os.listdir(os.path.join(GOPRO_PATH, prefix, "target")) if name in s]
    save_dir = os.path.join(GOPRO_PATH, prefix, "events")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ori_ids, tar_ids = sorted(ori_ids), sorted(tar_ids)
    steps = len(ori_ids) // len(tar_ids)
    assert len(ori_ids) == len(tar_ids) * steps, (prefix, name, len(ori_ids), len(tar_ids))
    if show_bar:
        iter = trange(len(tar_ids), file=sys.stdout)
    else:
        iter = range(len(tar_ids))
    for i in iter:
        save_id = tar_ids[i]
        im_paths = [os.path.join(GOPRO_ORI_PATH, prefix, name, "%06d.png" % ori_id) for ori_id in
                    ori_ids[i * steps:i * steps + steps]]
        avi_path = os.path.join(save_dir, "{}-".format(name) + "%06d.avi" % save_id)
        events_path = avi_path.replace(".avi", ".txt")
        blurred_im_path = os.path.join(GOPRO_PATH, prefix, "input", name+"-"+"%06d.png" % save_id)
        init_sharp_im_path = avi_path.replace(".avi", ".png")
        ims_to_avi(im_paths, avi_path)
        avi_to_events(avi_path, events_path)
        events_to_im(events_path, steps, blurred_im_path, init_sharp_im_path)


if __name__ == '__main__':
    # num_cores = int(mp.cpu_count())
    num_cores = GPUS
    print("num cores: {}".format(num_cores))
    pool = mp.Pool(num_cores)
    params = [["train", name, False] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "train"))]
    params += [["test", name, False] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "test"))]
    for i, p in enumerate(params):
        if i % num_cores == 0:
            params[i][2] = True

    results = [pool.apply_async(convert, args=(t[0], t[1], t[2])) for t in params]
    results = [p.get() for p in results]
