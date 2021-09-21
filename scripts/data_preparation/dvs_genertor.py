import json
import multiprocessing as mp
import os
import pickle
# import sys
idx = 0
# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from aiisp_tool.utils.oss_helper import OSSHelper
from imageio import imread, imwrite
from tqdm import tqdm

# if write data to oss
WRITE_TO_OSS = True
OSS_PREFIX = "s3://lzh-share/stereo_blur_data/"
LOCAL_TO_OSS = lambda local_path : OSS_PREFIX+local_path.split("stereo_blur_data/")[-1]

# datasets path
GOPRO_ORI_PATH = "./datasets/GOPRO_Large/"
GOPRO_PATH = "./datasets/GoPro/"
STEREO_ORI_PATH = "./datasets/stereo_blur/"
STEREO_PATH = "./datasets/stereo_blur_data/"

# dvs setting
V2E_PATH = "./.v2e"
SLOMO_CHECKPOINT = "{}/.pretrain/SuperSloMo39.ckpt".format(V2E_PATH)
POS_THRES, NEG_THRES = .2, .2  # use v2e --dvs_params clean will overwrite the --pos_thres and --neg_thres to .2
APPEND_ARGS = "--disable_slomo"
SIZE = (1280, 720)
FPS = 960
STEPS = 16
COMMAND = "python3 {}/v2e.py " \
          "-i %(input)s " \
          "-o %(output_folder)s " \
          "--avi_frame_rate={} --overwrite --auto_timestamp_resolution --timestamp_resolution=.001 " \
          "--output_height 720 --output_width 1280  --dvs_params %(dvs_params)s --pos_thres={} --neg_thres={} " \
          "--dvs_emulator_seed=0 --slomo_model={} --no_preview --skip_video_output {} " \
          "--dvs_text=%(output)s > /dev/null 2>&1".format(V2E_PATH, FPS, POS_THRES, NEG_THRES, SLOMO_CHECKPOINT,
                                                          APPEND_ARGS)
# env setting 
GPU_NUM = 8
CPU_NUM = int(mp.cpu_count())
# CPU_NUM = 1


class DVS_Genertor():
    def __init__(self, pairs=None):
        self.pairs = pairs
        avi_to_events_core_num = GPU_NUM if len(APPEND_ARGS) == 0 else CPU_NUM
        self.PIPELINE = dict(
            sharps_to_blur=(DVS_Genertor._sharps_to_blur, CPU_NUM),
            sharps_to_avi=(DVS_Genertor._sharps_to_avi, CPU_NUM),
            avi_to_events=(DVS_Genertor._avi_to_events, avi_to_events_core_num),
            events_to_voxel=(DVS_Genertor._events_to_voxel, CPU_NUM),
            avi_to_voxel=(DVS_Genertor._avi_to_voxel, CPU_NUM),
        )

    def run(self, pipeline):
        for p in pipeline:
            assert p in self.PIPELINE.keys()
        if "avi_to_voxel" in pipeline:
            assert ("avi_to_events" not in pipeline) and ("events_to_voxel" not in pipeline), "not compatibility"
            global COMMAND
            COMMAND = COMMAND.replace("--dvs_text=%(output)s",
                            "--dvs_numpy=%(output)s --dvs_numpy_diff=%(diff)f --dvs_numpy_steps=%(steps)d")
        for p in pipeline:
            self._multiprocessing(self.PIPELINE[p][0], self.PIPELINE[p][1])

    def _multiprocessing(self, fn, num_cores):
        print("Process: {}".format(fn.__name__))
        pool = Pool(num_cores)
        results = [pool.apply_async(DVS_Genertor._son_process, args=(self.pairs, fn, num_cores, idx,)) for idx in
                   range(num_cores)]
        results = [p.get() for p in results]

    @staticmethod
    def _son_process(pairs, fn, num_cores, idx):
        start_id, stop_id = DVS_Genertor._get_start_id_and_stop_id(len(pairs), num_cores, idx)
        iter = tqdm(pairs[start_id:stop_id]) if start_id == 0 else pairs[start_id: stop_id]
        for pair in iter:
            if start_id == 0:
                print("", flush=True)
            fn(pair)

    @staticmethod
    def _get_path(pair, name):
        CONVERT_FN = dict(
            sharp_paths=lambda x: x["sharp_paths"],
            blur_path=lambda x: x["target_path"].replace("target", "input"),
            avi_path=lambda x: x["target_path"].replace("target", "events").replace("png", "avi"),
            clean_events_path=lambda x: x["target_path"].replace("target", "events").replace("png", "clean.txt"),
            noisy_events_path=lambda x: x["target_path"].replace("target", "events").replace("png", "noisy.txt"),
            clean_voxel_path=lambda x: x["target_path"].replace("target", "events").replace("png", "clean.npy"),
            noisy_voxel_path=lambda x: x["target_path"].replace("target", "events").replace("png", "noisy.npy"),
        )
        assert name in CONVERT_FN.keys()
        return CONVERT_FN[name](pair)

    @staticmethod
    def _sharps_to_blur(pair):
        sharp_paths = DVS_Genertor._get_path(pair, "sharp_paths")
        blur_im = np.zeros([SIZE[1], SIZE[0], 3], dtype=np.float32)
        for p in sharp_paths:
            im = imread(p).astype(np.float32) / 255.
            blur_im += np.power(im, 2.2) / len(sharp_paths)
        blur_im = np.power(blur_im, 1 / 2.2) * 255.
        blur_path = DVS_Genertor._get_path(pair, "blur_path")
        if WRITE_TO_OSS:
            helper = OSSHelper()
            blur_im = cv2.imencode(".png", blur_im[:, :, ::-1])[1].tostring()
            helper.upload(blur_im, LOCAL_TO_OSS(blur_path), "bin")
        else:
            imwrite(blur_path, blur_im.astype(np.uint8))

    @staticmethod
    def _sharps_to_avi(pair):
        out = cv2.VideoWriter(DVS_Genertor._get_path(pair, "avi_path"), cv2.VideoWriter_fourcc(*'DIVX'), FPS, SIZE)
        # out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'HDYV'), FPS, SIZE)
        for p in DVS_Genertor._get_path(pair, "sharp_paths"):
            out.write(cv2.imread(p))
        out.release()

    @staticmethod
    def _avi_to_events(pair):
        avi_path = DVS_Genertor._get_path(pair, "avi_path")
        clean_events_path = DVS_Genertor._get_path(pair, "clean_events_path")
        noisy_events_path = DVS_Genertor._get_path(pair, "noisy_events_path")
        tmp_dir = "/tmp/{}".format(os.getpid())
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPU_NUM) + COMMAND % {'input': avi_path,
                                                                                    'output_folder': tmp_dir,
                                                                                    'output': clean_events_path,
                                                                                    'dvs_params': "clean"}
        os.system(cmd)
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPU_NUM) + COMMAND % {'input': avi_path,
                                                                                    'output_folder': tmp_dir,
                                                                                    'output': noisy_events_path,
                                                                                    'dvs_params': "noisy"}
        os.system(cmd)

    @staticmethod
    def __events_to_voxel(pair, clean=True, remove_txt=True):
        assert STEPS % 2 == 0 and STEPS != 0, "steps should be even and im_num should be odd"
        events = np.zeros([STEPS, 2, SIZE[1], SIZE[0]]).astype(np.uint16)
        im_num = len(DVS_Genertor._get_path(pair, "sharp_paths"))
        diff = float(STEPS) * float(im_num - 1) / float(FPS)
        dvs_params = "clean" if clean else "noisy"
        events_path = DVS_Genertor._get_path(pair, "{}_events_path".format(dvs_params))
        voxel_path = DVS_Genertor._get_path(pair, "{}_voxel_path".format(dvs_params))
        with open(events_path, "r+") as f:
            lines = (i for i in f.readlines() if not i.startswith("#"))
        for l in lines:
            _l = [float(i) for i in l.split("\n")[0].split(" ")]
            t, x, y, p = _l[0], int(_l[1]), int(_l[2]), int(_l[3])
            if t >= diff * STEPS:
                continue
            events[int(np.floor(t / diff)), p, y, x] += 1
        # maybe we need add a events denoising function here?
        events = events.astype(np.float32)
        events = events[:, 1] * POS_THRES - events[:, 0] * NEG_THRES
        print(events.max(), events.min(), flush=True)
        if WRITE_TO_OSS:
            helper = OSSHelper()
            helper.upload(events, LOCAL_TO_OSS(voxel_path), "numpy")
        else:
            np.save(voxel_path, events)

        if remove_txt:
            os.remove(events_path)

    @staticmethod
    def _events_to_voxel(pair):
        DVS_Genertor.__events_to_voxel(pair, clean=True)
        DVS_Genertor.__events_to_voxel(pair, clean=False)

    @staticmethod
    def _avi_to_voxel(pair):
        assert STEPS % 2 == 0 and STEPS != 0, "steps should be even and im_num should be odd"
        frames = len(DVS_Genertor._get_path(pair, "sharp_paths"))
        diff = 1. / float(FPS) * (frames - 1) / STEPS

        avi_path = DVS_Genertor._get_path(pair, "avi_path")
        clean_voxel_path = DVS_Genertor._get_path(pair, "clean_voxel_path")
        noisy_voxel_path = DVS_Genertor._get_path(pair, "noisy_voxel_path")
        if WRITE_TO_OSS:
            clean_voxel_path = LOCAL_TO_OSS(clean_voxel_path)
            noisy_voxel_path = LOCAL_TO_OSS(noisy_voxel_path)
        tmp_dir = "/tmp/{}".format(os.getpid())
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPU_NUM) + COMMAND % {'input': avi_path,
                                                                                    'output_folder': tmp_dir,
                                                                                    'output': clean_voxel_path,
                                                                                    'diff': diff,
                                                                                    'steps': STEPS,
                                                                                    'dvs_params': "clean"}
        print("aaaa", flush=True)
        os.system(cmd)
        print("bbbb", flush=True)
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(os.getpid() % GPU_NUM) + COMMAND % {'input': avi_path,
                                                                                    'output_folder': tmp_dir,
                                                                                    'output': noisy_voxel_path,
                                                                                    'diff': diff,
                                                                                    'steps': STEPS,
                                                                                    'dvs_params': "noisy"}
        os.system(cmd)

    @staticmethod
    def _get_start_id_and_stop_id(data_num, core_num, idx=None):
        idx = os.getpid() % core_num if idx is None else idx
        start_id, stop_id = data_num // core_num * idx, data_num // core_num * (idx + 1)
        if idx == core_num - 1:
            stop_id = data_num
        return start_id, stop_id


def stereo_generate_pairs():
    pairs_save_path = os.path.join(STEREO_PATH, "train_test_pairs.pkl")
    if os.path.exists(pairs_save_path):
        with open(pairs_save_path, "rb+") as f:
            return pickle.load(f)

    train_dir = os.path.join(os.getcwd(), STEREO_PATH, "train")
    test_dir = os.path.join(os.getcwd(), STEREO_PATH, "test")
    dir_list = [train_dir, test_dir]
    dir_list += [os.path.join(train_dir, __dir) for __dir in ["input", "target", "events"]]
    dir_list += [os.path.join(test_dir, __dir) for __dir in ["input", "target", "events"]]
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

    train_test_split = {}
    with open(os.path.join(STEREO_ORI_PATH, "stereo_deblur_data.json"), "r+") as f:
        for j in json.load(f):
            train_test_split[j["name"]] = True if j["phase"] == "Train" else False
    paths = [str(s) for s in Path(STEREO_ORI_PATH).glob("*/image_left_x16/*.png")]
    train_counter, test_counter = 0, 0
    pairs = []
    for p in tqdm(paths):
        is_train, idx = train_test_split[p.split("/")[-3]], int(float(p.split("/")[-1][:-4]))
        input_list, sharp_list = [], []
        for step in [17, 33, 49]:
            if idx < step + 16 or idx % step != 0:
                continue
            input_list.append([os.path.join(os.getcwd(), p[:-10], "%05d.png" % i) for i in range(idx - step, idx)])
            sharp_list.append(input_list[-1][step // 2])
        if is_train:
            out_paths = [os.path.join(train_dir, "target", "left_%07d.png" % (i + train_counter)) for i in
                         range(len(input_list))]
            train_counter += len(input_list)
        else:
            out_paths = [os.path.join(test_dir, "target", "left_%07d.png" % (i + test_counter)) for i in
                         range(len(input_list))]
            test_counter += len(input_list)
        for s, o in zip(sharp_list, out_paths):
            # copyfile(s, o)
            os.symlink(s, o)
        pairs += [dict(sharp_paths=inputs, target_path=out_path) for inputs, out_path in zip(input_list, out_paths)]

        input_list = [[p.replace("left", "right") for p in inputs] for inputs in input_list]
        sharp_list = [p.replace("left", "right") for p in sharp_list]
        out_paths = [p.replace("left", "right") for p in out_paths]
        for s, o in zip(sharp_list, out_paths):
            # copyfile(s, o)
            os.symlink(s, o)
        pairs += [dict(sharp_paths=inputs, target_path=out_path) for inputs, out_path in zip(input_list, out_paths)]

    with open(pairs_save_path, "wb+") as f:
        pickle.dump(pairs, f)
    return pairs


def gopro_generate_pairs():
    pairs_save_path = os.path.join(GOPRO_PATH, "train_test_pairs.pkl")
    if os.path.exists(pairs_save_path):
        with open(pairs_save_path, "rb+") as f:
            return pickle.load(f)

    pairs = []
    train_set = [("train", name) for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "train"))]
    test_set = [("test", name) for name in os.listdir(os.path.join(GOPRO_ORI_PATH, "test"))]
    for prefix, name in train_set + test_set:
        if not os.path.exists(os.path.join(GOPRO_PATH, prefix, "events")):
            os.makedirs(os.path.join(GOPRO_PATH, prefix, "events"))
        ori_ids = [int(float(str(s)[:-len(".png")])) for s in os.listdir(os.path.join(GOPRO_ORI_PATH, prefix, name))]
        tar_ids = [int(float(str(s)[len(name + "-"):-len(".png")])) for s in
                   os.listdir(os.path.join(GOPRO_PATH, prefix, "target")) if name in s]
        ori_ids, tar_ids = sorted(ori_ids), sorted(tar_ids)
        steps = len(ori_ids) // len(tar_ids)
        assert len(ori_ids) == len(tar_ids) * steps, (prefix, name, len(ori_ids), len(tar_ids))
        for i in range(len(tar_ids)):
            sharp_paths = [os.path.join(GOPRO_ORI_PATH, prefix, name, "%06d.png" % ori_id) for ori_id in
                           ori_ids[i * steps:i * steps + steps]]
            target_path = os.path.join(GOPRO_PATH, prefix, "target", name + "-" + "%06d.png" % tar_ids[i])
            pairs.append(dict(
                sharp_paths=sharp_paths,
                target_path=target_path
            ))

    with open(pairs_save_path, "wb+") as f:
        pickle.dump(pairs, f)
    return pairs


if __name__ == '__main__':
    stereo_pairs = stereo_generate_pairs()
    # idx = int(sys.argv[1])
    start_id = idx * len(stereo_pairs) // 8
    stop_id = (idx + 1) * len(stereo_pairs) // 8
    if len(stereo_pairs) - stop_id <= 8:
        stop_id = len(stereo_pairs)
    dvs_genertor = DVS_Genertor(stereo_pairs[start_id:stop_id])
    # dvs_genertor.run(["sharps_to_blur", "sharps_to_avi", "avi_to_events", "events_to_voxel"])
    dvs_genertor.run(["avi_to_voxel"])
    # gopro_pairs = gopro_generate_pairs()
