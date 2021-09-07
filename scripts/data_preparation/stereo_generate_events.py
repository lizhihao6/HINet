import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from imageio import imread, imwrite
from tqdm import tqdm, trange

STEREO_ORI_PATH = "./datasets/stereo_blur/"
STEREO_PATH = "./datasets/stereo_blur_data/"
V2E_PATH = "./.v2e"
SLOMO_CHECKPOINT = "{}/.pretrain/SuperSloMo39.ckpt".format(V2E_PATH)
POS_THRES, NEG_THRES = .2, .2  # use v2e --dvs_params clean will overwrite the --pos_thres and --neg_thres to .2
FPS = 480
SIZE = (1280, 720)
GPU_NUM = 8
CPU_NUM = int(mp.cpu_count())

COMMAND = "python3 {}/v2e.py " \
          "-i %(input)s " \
          "-o /tmp/output/$(date) " \
          "--avi_frame_rate=120 --overwrite --auto_timestamp_resolution --timestamp_resolution=.001 " \
          "--output_height 720 --output_width 1280  --dvs_params noisy --pos_thres={} --neg_thres={} " \
          "--dvs_emulator_seed=0 --slomo_model={} --no_preview " \
          "--dvs_text=%(output)s > /dev/null 2>&1".format(V2E_PATH, POS_THRES, NEG_THRES, SLOMO_CHECKPOINT)


def sharp_to_blur(im_paths, save_path):
    blur_im = np.zeros([SIZE[1], SIZE[0], 3])
    for p in im_paths:
        im = imread(p).astype(np.float32) / 255.
        blur_im += np.power(im, 2.2) / len(im_paths)
    imwrite(save_path, np.power(blur_im, 1 / 2.2))


def ims_to_avi(im_paths, save_path):
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, SIZE)
    # out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'HDYV'), FPS, SIZE)
    for p in im_paths:
        out.write(cv2.imread(p))
    out.release()


def avi_to_events(avi_path, save_path):
    cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPU_NUM) + COMMAND % {'input': avi_path, 'output': save_path}
    os.system(cmd)


def events_to_voxel(events_path, im_num, blurred_im_path, save_path):
    assert im_num % 2 != 0, "im_num should be odd"
    blurred_im = imread(blurred_im_path).astype(np.float32) / 255.
    linear_blurred_im = np.power(blurred_im, 2.2)
    y_blurred_im = linear_blurred_im * np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape([1, 1, 3])
    y_blurred_im = y_blurred_im.sum(2) * 255. + 16.

    events = np.zeros([im_num - 1, 2, y_blurred_im.shape[0], y_blurred_im.shape[1]]).astype(np.uint16)
    stack_events = np.zeros([2, y_blurred_im.shape[0], y_blurred_im.shape[1]]).astype(np.uint16)
    diff, half_time = 1. / float(FPS), float(im_num // 2) / float(FPS)
    with open(events_path, "r+") as f:
        lines = [i for i in f.readlines() if not i.startswith("#")]
    for l in lines:
        _l = [float(i) for i in l.split("\n")[0].split(" ")]
        t, x, y, p = _l[0], int(_l[1]), int(_l[2]), int(_l[3])
        if t > diff * (im_num - 1):
            continue
        if t < half_time:
            start_id, stop_id = 0, int(np.ceil(t / diff))
        else:
            start_id, stop_id = int(np.floor(t / diff)), im_num - 1
        stack_events[p, y, x] += 1
        for e in range(start_id, stop_id):
            _p = 1 - p if t < half_time else p
            events[e, _p, y, x] += 1
    stack_events, events = stack_events.astype(np.float32), events.astype(np.float32)
    stack_events = stack_events[1] * POS_THRES - stack_events[0] * NEG_THRES
    events = events[:, 1] * POS_THRES - events[:, 0] * NEG_THRES
    events = np.exp(events).sum(0) / (im_num - 1)
    assert events.shape == y_blurred_im.shape, "events shape should be consistent with blurred image shape"
    y_sharp_im = y_blurred_im / events
    y_sharp_im = np.power(np.clip(y_sharp_im, 16, 235) / 255., 1 / 2.2) * 255.
    imwrite(save_path, y_sharp_im.astype(np.uint8))
    np.save(save_path.replace(".png", ".npy"), stack_events)


def generate_pairs(pairs):
    pairs_save_path = os.path.join(STEREO_PATH, "train_test_pairs.pkl")
    if os.path.exists(pairs_save_path):
        with open(pairs_save_path, "rb+") as f:
            return pickle.load(f)
    train_test_split = {}
    for j in json.load(os.path.join(STEREO_PATH, "stereo_deblur_data.json")):
        train_test_split[j["name"]] = True if j["phase"] == "Train" else False
    paths = [str(s) for s in Path(STEREO_ORI_PATH).glob("*/image_*_x8/*.png")]
    train_counter, test_counter = 0, 0
    pairs = []
    for p in paths:
        is_train, idx = train_test_split[p.split("/")[-3]], int(float(p.split("/")[-1][:-4]))
        input_list, sharp_list = [], []
        for step in [17, 33, 49]:
            if idx < step or idx % step != 0:
                continue
            input_list.append([os.path.join(os.getcwd(), p[:-10], "%05d.png" % i) for i in range(idx - step, idx)])
            sharp_list.append(input_list[-1][step // 2])
        out_dir = os.path.join(os.getcwd(), STEREO_PATH, "train") if is_train else os.path.join(STEREO_PATH, "test")
        if is_train:
            out_paths = [os.path.join(out_dir, "target", "%07d.png" % (i + train_counter)) for i in
                         range(len(input_list))]
            train_counter += len(input_list)
        else:
            out_paths = [os.path.join(out_dir, "target", "%07d.png" % (i + test_counter)) for i in
                         range(len(input_list))]
            test_counter += len(input_list)
        pairs.append(dict(inputs=inputs, target=out_path) for inputs, out_path in zip(input_list, out_paths))

    with open(pairs_save_path, "wb+") as f:
        f.write(pairs)
    return pairs


def _get_start_id_and_stop_id(data_num, core_num):
    idx = os.getpid() % core_num
    start_id, stop_id = data_num // core_num * idx, data_num // core_num * (idx + 1)
    if idx == core_num - 1:
        stop_id = data_num
    return start_id, stop_id


def _get_paths(target_path):
    input_path = target_path.replace("target", "input")
    avi_path = target_path.replace("target", "events").replace("png", "avi")
    events_path = avi_path.replace("avi", "txt")
    aps_path = events_path.replace("txt", "png")
    voxel_path = events_path.replace("txt", "npy")
    return dict(
        input_path=input_path,
        avi_path=avi_path,
        events_path=events_path,
        aps_path=aps_path,
        voxel_path=voxel_path
    )


def convert_pairs_to_avi(pairs):
    start_id, stop_id = _get_start_id_and_stop_id(len(pairs), core_num=CPU_NUM)
    iter = tqdm(pairs[start_id, stop_id]) if start_id == 0 else pairs[start_id, stop_id]
    for i in iter:
        paths = _get_paths(i)
        sharp_to_blur(i["inputs"], i["target"])
        sharp_to_blur(i["inputs"], i["target"].replace("target", "input"))
        ims_to_avi(i["inputs"], i["target"].replace("target", "input").replace(".png", ".avi"))


def convert_avi_to_events(pairs):
    start_id, stop_id = _get_start_id_and_stop_id(len(pairs), core_num=GPU_NUM)
    iter = tqdm(pairs[start_id, stop_id]) if start_id == 0 else pairs[start_id, stop_id]
    for i in iter:
        avi_path = i["target"].repace
        avi_to_events()

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
        blurred_im_path = os.path.join(GOPRO_PATH, prefix, "input", name + "-" + "%06d.png" % save_id)
        init_sharp_im_path = avi_path.replace(".avi", ".png")
        ims_to_avi(im_paths, avi_path)
        avi_to_events(avi_path, events_path)
        events_to_im(events_path, steps, blurred_im_path, init_sharp_im_path)


if __name__ == '__main__':
    # num_cores = int(mp.cpu_count())
    num_cores = GPU_NUM
    print("num cores: {}".format(num_cores))
    pool = mp.Pool(num_cores)
    params = [["train", name, False] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "train"))]
    params += [["test", name, False] for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "test"))]
    for i, p in enumerate(params):
        if i % num_cores == 0:
            params[i][2] = True

    results = [pool.apply_async(convert, args=(t[0], t[1], t[2])) for t in params]
    results = [p.get() for p in results]
