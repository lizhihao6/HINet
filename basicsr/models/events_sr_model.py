# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import os
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import numpy as np
import sparseconvnet as scn
import torch
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pytorch3d.ops import knn_points, knn_gather
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from matplotlib import pyplot as plt
import emd_linear.emd_module as emd
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class TransSRModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(TransSRModel, self).__init__(opt)
        self.voxel = opt['voxel']
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.emd = emd.emdModule()

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        self.classes = self.opt.get('classes', None)
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        # train_opt = self.opt['train']
        # define losses
        if not self.voxel:
            self.bce = torch.nn.BCELoss()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(self.net_g.parameters(), **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        [setattr(self, k, data[k].to(self.device)) for k in data.keys() if k != 'key']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.voxel:
            self.output = self.net_g(self.lq)
        else:
            self.output, bias = self.net_g(self.lq)
        loss_dict = OrderedDict()

        if self.voxel:
            l_total = self.mse(self.output, self.gt)
        else:
            b, n, c = self.output.shape
            assert c == 3
            assert n == self.gt.shape[1]
            idx = knn_points(self.output, self.gt[:, :, :3], K=1).idx
            gt_xyz = knn_gather(self.gt, idx)[:, :, 0, :3]
            baselines = self.l1(self.output, gt_xyz)
            loss_dict['baselines'] = baselines

            l2_norm = sum(p.pow(2.0).sum() for p in self.net_g.parameters())
            loss_dict['l2_norm'] = l2_norm

            xyz = self.output[:, :, :3]+bias
            dist, _ = self.emd(xyz, self.gt[:, :, :3], 0.005, 50)
            emd = torch.sqrt(dist).mean(1).mean()
            loss_dict['emd'] = emd

            # l_total = emd*100 + l2_norm
            l_total = emd

        loss_dict['l_total'] = l_total
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', False)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def split(self):
        '''

        Returns:

            self.ori_lq = lq

            voxel:
                self.frame_start_ids, self.frame_stop_ids, self.lq=split_voxel
            point:
                self.start_timestamp, self.stop_timestamp, self.lq = split_points

        '''
        self.origin_lq = self.lq
        if self.voxel:
            b, n, _, _ = self.lq.size()
            assert b == 1
            crop_size = self.opt['network_g']['in_ch']
            parts = []
            start_ids, stop_ids = [], []
            for i in range(0, n - crop_size, crop_size):
                start_id = i if i + crop_size < n else n - crop_size
                part_frames = self.lq[:, start_id:start_id + crop_size]
                parts.append(part_frames)
                start_ids.append(start_id)
                stop_ids.append(min(start_id + crop_size, n - 1))  # exclude
            self.frame_start_ids, self.frame_stop_ids = start_ids, stop_ids
            self.lq = torch.cat(parts, dim=0)
            del parts[:]
        else:
            b, n, _ = self.lq_events.size()
            assert b == 1
            crop_size = self.opt['point_num']
            parts = []
            start_ids, stop_ids = [], []
            for i in range(0, n - crop_size, crop_size):
                start_id = i if i + crop_size < n else n - crop_size
                part_events = self.lq_events[:, start_id:start_id + crop_size]
                part_events[:, :, 2] = (part_events[:, :, 2] - part_events[:, 0, 2]) / (
                        part_events[:, -1, 2] - part_events[:, 0, 2])
                parts.append(part_events)
                start_ids.append(start_id)
                stop_ids.append(min(start_id + crop_size, n - 1))  # exclude
            self.lq = torch.cat(parts, dim=0)
            del parts[:]
            gt_event_start_ids = self.map_list[0, start_ids]
            gt_event_stop_ids = self.map_list[0, stop_ids]  # exclude
            gt_start_timestamp = self.gt_events[0, gt_event_start_ids, 2]
            gt_stop_timestamp = self.gt_events[0, torch.clamp(gt_event_stop_ids, max=self.gt_events.shape[1]), 2]
            self.start_timestamp, self.stop_timestamp = gt_start_timestamp, gt_stop_timestamp

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if self.voxel:
                    outs.append(pred)
                else:
                    pred = torch.clamp(pred, 0, 1)
                    pred[:, :, 0] = pred[:, :, 0]
                    pred[:, :, 1] = pred[:, :, 1]
                    gt_start_time, gt_stop_time = self.start_timestamp[i:j][:, None], self.stop_timestamp[i:j][:, None]
                    pred[:, :, 2] = pred[:, :, 2] * (gt_stop_time - gt_start_time) + gt_start_time
                    pred[:, :, 3] = torch.round(pred[:, :, 3])
                    # for increase sort
                    idx = pred[:, :, 2].sort(1).indices
                    outs += [p[0, idx[pi]].unsqueeze(0) for pi, p in enumerate(torch.split(pred, 1, dim=0))]
                i = j

        if self.voxel:
            diff = self.frame_start_ids[-1] - self.frame_stop_ids[-2]
            outs[-1] = outs[-1][diff:]  # len(outs) = total_frame
            self.output = torch.cat(outs, dim=0).reshape(1, -1, outs[-1].shape[-2],
                                                         outs[-1].shape[-1])  # [1, total_frame, h, w]
        else:
            # remove duplicate
            last_out_timestamps = outs[-1][0, :, 2]
            mask = torch.where(last_out_timestamps > self.stop_timestamp[-2], last_out_timestamps,
                               torch.zeros_like(last_out_timestamps))
            outs[-1] = outs[-1][:, torch.nonzero(mask, as_tuple=True)[0], :]
            self.output = torch.cat(outs, dim=1)  # [1, total_events, 4]
            self.output[:, :, 2:] = torch.round(self.output[:, :, 2:])
        del outs[:]
        self.net_g.train()

    def split_inverse(self):
        self.lq = self.origin_lq
        if self.voxel:
            self.output_voxel = self.output
            self.output_events = None
            assert self.output_voxel.shape[1] == self.lq.shape[1]
        else:
            self.output_voxel = None
            self.output_events = self.output

    def _events_to_voxel(self, events, h, w, f, time_resolution=10000):
        timestamps = events[0, :, 2]
        gt_timestamps = self.gt_events[0, :, 2]
        events_frames = torch.zeros([1, f, h, w]).to(events.device)
        events_to_frame = scn.Sequential(
            scn.InputLayer(dimension=2, spatial_size=torch.LongTensor([int(h), int(w)]), mode=3),  # sum
            scn.SparseToDense(2, 1)
        )

        start_ids, stop_ids = [], []
        pointer = 0
        cache = timestamps.shape[0] // f * 10
        for i, t in enumerate(range(int(gt_timestamps[0]) + time_resolution, int(gt_timestamps[-1]), time_resolution)):
            diff = torch.abs(timestamps[pointer:pointer + cache] - float(t))
            _pointer = torch.argmin(diff) + pointer

            if timestamps[_pointer] > t:
                _pointer -= 1
            if (pointer != 0 and _pointer == pointer) or _pointer < 0:  # no data in this frame
                start_ids.append(pointer)
                stop_ids.append(_pointer)
                continue
            assert pointer < _pointer < pointer + cache or pointer == 0 or pointer == timestamps.shape[0], 'cache small'

            part_events = events[:, pointer:_pointer, :]
            y = part_events[0, :, 1][:, None] * h
            x = part_events[0, :, 0][:, None] * w
            y, x = torch.round(y), torch.round(x)
            b = torch.zeros([part_events.shape[1], 1]).to(part_events.device)
            yxb = torch.cat([y, x, b], dim=1).long()
            p = part_events[0, :, 3][:, None]
            p = (p - 0.5) / 0.5
            events_frames[0, i] = events_to_frame([yxb, p])[0, 0]
            start_ids.append(pointer)
            stop_ids.append(_pointer)

            pointer = _pointer

        if pointer < events.shape[1]:
            part_events = events[:, pointer:, :]
            y = part_events[0, :, 1][:, None] * h
            x = part_events[0, :, 0][:, None] * w
            y, x = torch.round(y), torch.round(x)
            b = torch.zeros([part_events.shape[1], 1]).to(part_events.device)
            yxb = torch.cat([y, x, b], dim=1).long()
            p = part_events[0, :, 3][:, None]
            p = (p - 0.5) / 0.5
            events_frames[0, -1] = events_to_frame([yxb, p])[0, 0]
            start_ids.append(pointer)
            stop_ids.append(events.shape[1])

        return events_frames, torch.LongTensor(start_ids), torch.LongTensor(stop_ids)

    def _generate_maplist(self, events, gt_events):
        b, n, _ = events.shape
        _, N, _ = gt_events.shape
        assert b == 1
        step, ratio = 64, 40
        map_list = torch.zeros([n]).long()
        gt_start_id = 0
        for i in range(0, n, step):
            lq_start_id, lq_stop_id = i, min(i + step, n)
            gt_stop_id = min(gt_start_id + step * ratio, N)
            lq_ts = events[0, lq_start_id:lq_stop_id, 2][:, None]  # [lq_n, 1]
            gt_ts = gt_events[:, gt_start_id:gt_stop_id, 2]  # [1, gt_N]
            index = gt_start_id + torch.argmin(torch.abs(lq_ts - gt_ts), dim=1)  # [lq_n]
            map_list[lq_start_id:lq_stop_id] = index
            assert index[-1] == index.max()
            gt_start_id = int(index[-1])
        return map_list

    def single_image_inference(self, todo):
        raise NotImplementedError

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        import os
        rank = os.environ['LOCAL_RANK']
        use_pbar = True if rank == '0' else False
        metric = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image,
                                         print_log=False, use_pbar=use_pbar)

        torch.distributed.reduce(metric['mse'], dst=0)
        torch.distributed.reduce(metric['mse_counter'], dst=0)
        if not self.voxel:
            torch.distributed.reduce(metric['cd'], dst=0)
            torch.distributed.reduce(metric['cd_counter'], dst=0)
        if rank == '0':
            self.metric_results['mse'] = float(metric['mse'] / metric['mse_counter'])
            self.metric_results['cd'] = float(metric['cd'] / metric['cd_counter'])
            dataset_name = dataloader.dataset.opt['name']
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            return self.metric_results['mse']
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image, use_pbar=True, print_log=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='events')

        mse, cd = 0., 0.
        mse_counter, cd_counter = 0, 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['key'][0]))[0]

            self.feed_data(val_data)
            # lq -> split lq_frame lq_events -> split lq_events
            if use_pbar:
                pbar.set_description('Splitting data...')
            self.split()
            if use_pbar:
                pbar.set_description('Testing data...')
            self.test()
            # self.output -> self.output_events, self.output_voxel
            if use_pbar:
                pbar.set_description('Combining data...')
            self.split_inverse()
            if not self.voxel:
                b, f, h, w = self.gt.shape
                assert b == 1
                if use_pbar:
                    pbar.set_description('Convert events to voxel...')
                self.output_voxel, start_ids, stop_ids = self._events_to_voxel(self.output_events, h, w, f)
                if use_pbar:
                    pbar.set_description('Mapping events...')
                map_list = self._generate_maplist(self.output_events, self.gt_events)
                crop_size = self.opt['val']['cd_input_num']

                for i in range(0, self.output_events.shape[1], crop_size):
                    start_id, stop_id = i, min(i + crop_size, self.output_events.shape[1])  # exclude
                    events = self.output_events[:, start_id: stop_id].float()
                    events[:, :, 0] = events[:, :, 0] / w
                    events[:, :, 1] = events[:, :, 1] / h
                    events[:, :, 2] = (events[:, :, 2] - events[:, 0, 2]) / (events[:, -1, 2] - events[:, 0, 2])
                    start_id, stop_id = map_list[start_id], map_list[stop_id - 1] + 1
                    gt_events = self.gt_events[:, start_id:stop_id]
                    # the last 1024 points may not have corresponding gt points
                    if start_id == stop_id:
                        continue
                    gt_events[:, :, 2] = (gt_events[:, :, 2] - gt_events[:, 0, 2]) / (
                            gt_events[:, -1, 2] - gt_events[:, 0, 2])
                    _cd, _ = chamfer_distance(events, gt_events)
                    cd += _cd.data * crop_size
                    cd_counter += crop_size

            mse += self.mse(self.output_voxel, self.gt) * self.output_voxel.shape[1]
            mse_counter += self.output_voxel.shape[1]

            if save_img:
                if use_pbar:
                    pbar.set_description('Save images...')

                def _save_img(path_str, k, frame, v_min, v_max):
                    frame = frame.sum(0).detach().cpu().numpy()  # [h, w]
                    fig, ax = plt.subplots()
                    ax.set_axis_off()
                    # draw background
                    ax.imshow(np.ones_like(frame), cmap='gray', vmin=0, vmax=1)
                    # draw positive
                    pos = np.where(frame > 0., frame, np.zeros_like(frame))
                    alpha = np.where(pos > 0, np.ones_like(frame), np.zeros_like(frame))
                    ax.imshow(pos, alpha=alpha, cmap='bwr', vmin=-v_max, vmax=v_max)
                    # draw negtive
                    neg = np.where(frame < 0., frame, np.zeros_like(frame))
                    alpha = np.where(neg < 0, np.ones_like(frame), np.zeros_like(frame))
                    ax.imshow(neg, alpha=alpha, cmap='bwr', vmin=v_min, vmax=-v_min)

                    save_dir = '/'.join(path_str.split('/')[:-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    fig.savefig(path_str.format(k), bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                def _save_events(path_str, k, events):
                    events = events.detach().cpu().numpy()  # [n, 4]
                    if events[-1, 2] == events[0, 2]:
                        events[:, 2] = 0
                    else:
                        events[:, 2] = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2])
                    np.save(path_str.format(k + '_events'), events)

                frames = self.opt['val']['frames']
                frame_num = self.output_voxel.shape[1]
                for i in range(frame_num - frames):
                    if i % self.opt['val']['save_freq'] != 0:
                        continue
                    if self.opt['is_train']:
                        path_str = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_' + '{}' + f'_{i}_{current_iter}.png')
                    else:
                        path_str = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_' + '{}' + f'_{i}.png')
                    lq, output, gt = self.lq[0, i:i + frames], self.output_voxel[0, i:i + frames], self.gt[0, i:i + frames]
                    v_min = min(lq.sum(0).min(), gt.sum(0).min())
                    v_max = max(lq.sum(0).max(), gt.sum(0).max())
                    _save_img(path_str, 'lq', lq, v_min, v_max)
                    _save_img(path_str, 'output', output, v_min, v_max)
                    _save_img(path_str, 'gt', gt, v_min, v_max)

                    if not self.voxel and self.opt['val']['save_events']:
                        path_str = path_str.replace('png', 'npy')
                        start_id, stop_id = self.lq_frame_start_ids[0, i], self.lq_frame_stop_ids[0, i + frames]
                        if start_id != stop_id:
                            _save_events(path_str, 'lq', self.lq_events[0, start_id:stop_id])
                        start_id, stop_id = start_ids[i], stop_ids[i + frames]
                        if start_id != stop_id:
                            _save_events(path_str, 'lq', self.output_events[0, start_id:stop_id])
                        start_id, stop_id = self.gt_frame_start_ids[0, i], self.gt_frame_stop_ids[0, i + frames]
                        if start_id != stop_id:
                            _save_events(path_str, 'lq', self.gt_events[0, start_id:stop_id])

            del self.lq, self.gt, self.output_events, self.output_voxel, self.output
            if not self.voxel:
                del self.lq_events, self.gt_events, self.map_list, start_ids, stop_ids
                del self.lq_frame_start_ids, self.lq_frame_stop_ids, self.gt_frame_start_ids, self.gt_frame_stop_ids

            torch.cuda.empty_cache()
            self.metric_results = {'mse': mse / mse_counter}
            if not self.voxel:
                self.metric_results['cd'] = cd / cd_counter

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            if print_log:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        device = self.net_g.device
        data = {'mse': torch.FloatTensor([mse]).to(device),
                'mse_counter': torch.LongTensor([mse_counter]).to(device),
                'cd': torch.FloatTensor([cd]).to(device),
                'cd_counter': torch.LongTensor([cd_counter]).to(device)}
        return data

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'

        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        raise NotImplementedError

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
