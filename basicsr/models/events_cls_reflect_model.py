# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from pytorch3d.ops import add_points_features_to_volume_densities_features
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class TransClsReflectModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(TransClsReflectModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # print('Params:', sum(p.numel() for p in self.net_g.parameters() if p.requires_grad))
        # from thop import profile, clever_format
        # inp = torch.rand(1, 1024, 4).cuda()
        # # Count the number of FLOPs
        # macs, params = profile(self.net_g, inputs=(inp, ))
        # macs, params = clever_format([macs, params], "%.3f")
        # print(macs, params)
        # exit(-1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        self.classes = self.opt.get('classes', None)

        if self.is_train:
            self.correct, self.counter = 0., 0.
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        # train_opt = self.opt['train']
        # define losses
        self.use_bce = self.opt['network_g']['cls_num'] == 2
        self.loss_fn = torch.nn.BCELoss() if self.use_bce else torch.nn.CrossEntropyLoss()

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
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, data):
        self.events = data['events'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def split(self):
        b, n, _ = self.events.size()
        assert b == 1
        crop_size = self.opt['val'].get('sample_num')
        assert n % crop_size == 0
        parts = []
        for i in range(0, n, crop_size):
            part_events = self.events[:, i:i + crop_size]
            part_events[:, :, 2] = (part_events[:, :, 2] - part_events[:, 0, 2]) / (
                    part_events[:, -1, 2] - part_events[:, 0, 2])
            parts.append(part_events)
        self.origin_events = self.events
        self.events = torch.cat(parts, dim=0)

    def split_inverse(self):
        # idx = torch.argmax(self.output, dim=1, keepdim=True) # [B]
        # self.output = torch.zeros([1, idx]).to(self.output.device)
        # for i, v in enumerate(idx):
        #     self.output[i] = v
        # print(self.output)
        if isinstance(self.output, list):
            self.output = sum([o for o in self.output])
        # _output = torch.mean(self.output, dim=0, keepdim=True)  # [1, classes_num]
        # if torch.argmax(_output, dim=1) != self.gt[0]:
        #     print(self.output, self.gt, self.key, self.origin_events.shape, self.output.shape)
        # self.output = _output
        self.output = torch.mean(self.output, dim=0, keepdim=True)  # [1, classes_num]
        events = self.origin_events
        events[:, :, 2] = (events[:, :, 2] - events[:, 0, 2]) / (events[:, -1, 2] - events[:, 0, 2])
        self.events = events

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.events)
        # if current_iter % 20 ==0:
        #     print(self.output[0, :].max(), torch.argmax(self.output[0]), self.gt[0])
        loss_dict = OrderedDict()
        if self.use_bce:
            self.gt = self.gt.float()
        if isinstance(self.output, list):
            l_total = sum([self.loss_fn(o, self.gt) for o in self.output])
        else:
            l_total = self.loss_fn(self.output, self.gt)
        if self.use_bce:
            self.output = self.output.unsqueeze(1).repeat(1, 2)
            self.output[:, 0] = 1-self.output[:, 0]
        loss_dict['l_cls'] = l_total
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward(retain_graph=True)
        # for name, parms in self.net_g.named_parameters():
        #     if 'module.backbone.embedding.0' in name:
        #         print(name)
        #         print(parms.grad.max(), parms.grad.min(), torch.abs(parms.grad).mean())

        use_grad_clip = self.opt['train'].get('use_grad_clip', False)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # calculate acc:
        if isinstance(self.output, list):
            if current_iter % 1000 == 0 or isinstance(self.correct, float):
                self.correct, self.counter = np.zeros([len(self.output) + 1]), 0.
            for i, o in enumerate(self.output):
                pred_choice = o.data.max(1)[1]
                self.correct[i] += pred_choice.eq(self.gt.long().data).cpu().sum()
            pred_choice = sum([o.data for o in self.output]).max(1)[1]
            self.correct[-1] += pred_choice.eq(self.gt.long().data).cpu().sum()
            self.counter += self.output[0].shape[0]
            for i, c in enumerate(self.correct):
                self.log_dict['acc_{}'.format(i)] = c / self.counter
        else:
            if current_iter % 1000 == 0:
                self.correct, self.counter = 0., 0.
            pred_choice = self.output.data.max(1)[1]
            self.correct += pred_choice.eq(self.gt.long().data).cpu().sum()
            self.counter += self.output.shape[0]
            self.log_dict['acc'] = self.correct / self.counter

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.events.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.events[i:j])
                outs.append(pred)
                i = j
            if isinstance(outs[0], list):
                self.output = [torch.cat([o[i] for o in outs], dim=0) for i in range(len(outs[0]))]
            else:
                self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def single_image_inference(self, events, save_path):
        self.feed_data(data={'events': events.unsqueeze(dim=0)})

        if self.opt['val'].get('split') is not None:
            self.split()

        self.test()

        if self.opt['val'].get('split') is not None:
            self.split_inverse()

        visuals = self.get_current_visuals()
        events = tensor2img([visuals['events']])
        imwrite(events, save_path)
        class_id = np.argmax(visuals['output'].numpy(), axis=1)[0]
        print('save_path: {} \n class_id: {} class: {}'.format(save_path, class_id, self.classes[class_id]))

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        import os
        rank = os.environ['LOCAL_RANK']
        use_pbar = True if rank == '0' else False
        metric = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image,
                                         print_log=False, use_pbar=use_pbar)
        metric = torch.FloatTensor([metric])[0].to(int(rank))
        data_len = torch.FloatTensor([dataloader.__len__()])[0].to(int(rank))
        metric *= data_len
        torch.distributed.reduce(metric, dst=0)
        torch.distributed.reduce(data_len, dst=0)
        if rank == '0':
            metric = float(metric / data_len)
            dataset_name = dataloader.dataset.opt['name']
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            return metric
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

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['key'][0]))[0]
            self.key = val_data['key']
            self.feed_data(val_data)
            if self.opt['val'].get('split') is not None:
                self.split()

            self.test()

            if self.opt['val'].get('split') is not None:
                self.split_inverse()

            visuals = self.get_current_visuals()
            events = tensor2img([visuals['events']])

            # tentative for out of GPU memory
            del self.events
            del self.output
            if 'gt' in visuals.keys():
                del self.gt
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                imwrite(events, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['output'], visuals['gt'], **opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
            cnt += 1
        if use_pbar:
            pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            if print_log:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

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
        out_dict = OrderedDict()
        events = (self.events.clone() - 0.5) / 0.5
        h, w = self.opt['val'].get('height', 300), self.opt['val'].get('width', 300)
        volume_densities = torch.zeros([events.shape[0], 1, 1, h, w]).to(events.device)
        volume_features = torch.zeros_like(volume_densities)
        volume_features, _ = add_points_features_to_volume_densities_features(events[:, :, :3],
                                                                              events[:, :, 3][:, :, None],
                                                                              volume_densities,
                                                                              volume_features)  # [B, 1, 1, h, w]
        events = volume_features[:, None, 0, 0].detach().cpu()  # [B, 1, h, w]
        out_dict['events'] = (events + 1) / 2
        if self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()  # [B]
        out_dict['output'] = self.output.detach().cpu()  # [B, classes_num]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
