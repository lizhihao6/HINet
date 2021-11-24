import numpy as np
import sparseconvnet as scn
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points

from basicsr.models.archs.arch_util import SwapAxes


class LocalTransformer(nn.Module):
    def __init__(self, feature_num, hidden_num, k_nearest) -> None:
        super().__init__()
        '''
        According to Point Transfomer Sec 3.2
        '''
        self.k_nearest = k_nearest

        self.kernel_fn = nn.Linear(feature_num, hidden_num)
        self.aggregation_fn = nn.Linear(hidden_num, feature_num)

        self.w_qs = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_ks = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_vs = nn.Linear(hidden_num, hidden_num, bias=False)

        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyzp, features):
        xyz = xyzp[:, :, :3]
        knn_result = knn_points(xyz, xyz, K=self.k_nearest)
        idx = knn_result.idx
        pos_enc = self.pos_encoder(xyzp[:, :, None] - knn_gather(xyzp, idx))  # [B, N, k_nearest, hidden_num]

        x = self.kernel_fn(features)
        # q: [B, N, hidden_num], k: [B, N, k_nearest, hidden_num], v: [B, N, k_nearest, hidden_num]
        q, k, v = self.w_qs(x), knn_gather(self.w_ks(x), idx), knn_gather(self.w_vs(x), idx)
        attention = self.attention_fn(q[:, :, None] - k + pos_enc)  # [B, N, k_nearest, hidden_num]
        attention = F.softmax(attention / np.sqrt(k.size(-1)), dim=-2)  # [B, N, k_nearest, hidden_num]

        res = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)
        res = self.aggregation_fn(res) + features
        # return res, attention
        return res


class ConvTransformer(nn.Module):
    def __init__(self, feature_num, hidden_num, k_nearest, h, w) -> None:
        super().__init__()
        self.k_nearest = k_nearest
        self.h, self.w = h, w

        self.norm = nn.LayerNorm(feature_num + 2)
        self.qkv = scn.Sequential(
            scn.InputLayer(dimension=2, spatial_size=torch.LongTensor([h, w]), mode=4),  # mean
            scn.SubmanifoldConvolution(dimension=2, nIn=feature_num + 2, nOut=hidden_num, filter_size=3, bias=True),
            scn.SubmanifoldConvolution(dimension=2, nIn=hidden_num, nOut=hidden_num * 3, filter_size=3, bias=False),
            scn.OutputLayer(hidden_num)
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.aggregation_fn = nn.Linear(hidden_num, feature_num)
        self.mlp = nn.Sequential(
            nn.Linear(feature_num, hidden_num),
            nn.GELU(),
            nn.Linear(hidden_num, feature_num)
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyzp, features):
        B, N = xyzp.shape[:2]
        # position encoding
        xy0 = self._get_xy0(xyzp)
        idx = knn_points(xy0, xy0, K=self.k_nearest).idx
        # [B, N, k_nearest, hidden_num]
        xyp = xyzp[:, :, [0, 1, 3]]
        pos_enc = self.pos_encoder(xyp[:, :, None] - knn_gather(xyp, idx))
        # get q, k, v
        yxb = self._get_yxb(xyzp)
        p = (xyzp[..., 3].view(B, N, 1) - 0.5) / 0.5  # [B, N, 1]
        pos = torch.clamp(p, 0, 1)
        neg = -torch.clamp(p, -1, 0)
        sparse = torch.cat([pos, neg, features], dim=-1)
        sparse = self.norm(sparse).reshape(B * N, -1)
        qkv = self.qkv([yxb, sparse]).reshape([B, N, -1, 3]).permute(3, 0, 1, 2)
        # q: [B, N, hidden_num], k: [B, N, k_nearest, hidden_num], v: [B, N, k_nearest, hidden_num]
        q, k, v = qkv[0], knn_gather(qkv[1], idx), knn_gather(qkv[2], idx)
        attention = self.attention_fn(q[:, :, None] - k + pos_enc)  # [B, N, k_nearest, hidden_num]
        attention = F.softmax(attention / np.sqrt(k.size(-1)), dim=-2)  # [B, N, k_nearest, hidden_num]

        res = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)
        res = self.aggregation_fn(res) + features
        res = self.mlp(res) + features
        return res

    def _get_yxb(self, xyzp):
        B, N = xyzp.shape[:2]
        if not hasattr(self, 'b') or self.b.shape[0] != B * N:
            self.b = torch.zeros([B * N, 1]).long().to(xyzp.device)
            for i in range(B):
                self.b[i * N:i * N + N] = i
        yx = xyzp[:, :, [1, 0]].view(-1, 2)
        yx[:, 0] = torch.round(yx[:, 0] * self.h)
        yx[:, 1] = torch.round(yx[:, 1] * self.w)
        yxb = torch.cat([yx, self.b], dim=-1).long()
        return yxb

    def _get_xy0(self, xyzp):
        B, N = xyzp.shape[:2]
        if not hasattr(self, 'z') or self.b.shape[0] != B * N:
            self.z = torch.zeros([B, N, 1]).long().to(xyzp.device)
        return torch.cat([xyzp[:, :, :2], self.z], dim=-1)


class GlobalTransfomer(nn.Module):
    def __init__(self, feature_num, hidden_num, k_nearest, global_point_num) -> None:
        super().__init__()
        self.k_nearest = k_nearest
        self.down = TransitionDown(global_point_num, k_nearest, feature_num, feature_num)

        self.norm = nn.LayerNorm(feature_num)

        self.kernel_fn = nn.Linear(feature_num, hidden_num)
        self.global_kernel_fn = nn.Linear(feature_num, hidden_num)
        self.aggregation_fn = nn.Linear(hidden_num, feature_num)

        self.w_qs = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_ks = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_vs = nn.Linear(hidden_num, hidden_num, bias=False)

        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyzp, features):
        features = self.norm(features)
        new_xyzp, new_features = self.down(xyzp, features)  # [B, global_point_num, 3], [B, global_point_num, F]
        xyz, new_xyz = xyzp[:, :, :3], new_xyzp[:, :, :3]
        knn_result = knn_points(xyz, new_xyz, K=self.k_nearest)
        idx = knn_result.idx
        pos_enc = self.pos_encoder(xyzp[:, :, None] - knn_gather(new_xyzp, idx))  # [B, N, k_nearest, hidden_num]

        x = self.kernel_fn(features)
        new_x = self.global_kernel_fn(new_features)
        # q: [B, N, hidden_num], k: [B, N, k_nearest, hidden_num], v: [B, N, k_nearest, hidden_num]
        q, k, v = self.w_qs(x), knn_gather(self.w_ks(new_x), idx), knn_gather(self.w_vs(new_x), idx)
        attention = self.attention_fn(q[:, :, None] - k + pos_enc)  # [B, N, k_nearest, hidden_num]
        attention = F.softmax(attention / np.sqrt(k.size(-1)), dim=-2)  # [B, N, k_nearest, hidden_num]

        res = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)
        res = self.aggregation_fn(res) + features
        return res


# class TransitionUp(nn.Module):
#     def __init__(self, in_ch_1, in_ch_2, out_ch, hidden_layer_num=2):
#         super().__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_ch_1, out_ch),
#             SwapAxes(),
#             nn.BatchNorm1d(out_ch),
#             SwapAxes(),
#             nn.ReLU(),
#         )
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_ch_2, out_ch),
#             SwapAxes(),
#             nn.BatchNorm1d(out_ch),
#             SwapAxes(),
#             nn.ReLU(),
#         ) if in_ch_2 != out_ch else nn.Identity()
#
#         self.mlp = nn.Sequential()
#         self.mlp.add_module('swap_in', SwapAxes())
#         for i in range(hidden_layer_num):
#             _in_ch = out_ch * 2 + 2 if i == 0 else out_ch
#             self.mlp.add_module('mlp_conv_{}'.format(i), nn.Conv1d(_in_ch, out_ch, 1))
#             # self.mlp.add_module('mlp_bn_{}'.format(i), nn.BatchNorm1d(out_ch))
#             self.mlp.add_module('relu_{}'.format(i), nn.ReLU())
#         self.mlp.add_module('swap_out', SwapAxes())
#
#     def forward(self, events1, features1, events2, features2):
#         p = (events1[..., 3][..., None] - 0.5) / 0.5  # [B, N_down, 1]
#         pos = torch.clamp(p, 0, 1)
#         neg = -torch.clamp(p, -1, 0)
#         features1 = torch.cat([pos, neg, self.fc1(features1)], dim=-1)  # [B, N_down, out_ch+2]
#         features2 = self.fc2(features2)  # [B, N_up, out_ch]
#
#         xyz1, xyz2 = events1[:, :, :3], events2[:, :, :3]
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_features = features2.repeat(1, N, 1)
#         else:
#             # dists [B, N_up, 3], idx [B, N_up, 3]
#             knn_result = knn_points(xyz2, xyz1, K=3)
#             dists, idx = knn_result.dists, knn_result.idx
#             dist_recip = 1.0 / (dists + 1e-8)  # [B, N_up, 3]
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)  # [B, N_up, 1]
#             weight = dist_recip / norm  # [B, N_up, 3]
#             interpolated_features = torch.sum(knn_gather(features1, idx) * weight.view(B, S, 3, 1), dim=2)
#
#         new_features = torch.cat([features2, interpolated_features], dim=-1)  # [B, N, F*2]
#         new_features = self.mlp(new_features)  # [B, N, F]
#         return new_features + features2


class TransitionDown(nn.Module):
    def __init__(self, out_point_num, k_nearest, in_ch, out_ch, hidden_layer_num=2, norm=True):
        super().__init__()
        self.out_point_num = out_point_num
        self.k_nearest = k_nearest
        self.mlp = nn.Sequential()
        for i in range(hidden_layer_num):
            _in_ch = in_ch + 3 if i == 0 else out_ch
            self.mlp.add_module('mlp_conv_{}'.format(i), nn.Conv2d(_in_ch, out_ch, 1))
            if norm:
                self.mlp.add_module('mlp_bn_{}'.format(i), nn.BatchNorm2d(out_ch))
            self.mlp.add_module('relu_{}'.format(i), nn.ReLU())

    def forward(self, xyzp, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, F]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_features_concat: sample points feature data, [B, S, D']
        """
        xyz = xyzp[:, :, :3]
        new_xyz, _ = sample_farthest_points(xyz, K=self.out_point_num)  # [B, out_points_num, 3]
        knn_result = knn_points(new_xyz, xyz, K=self.k_nearest)
        idx = knn_result.idx
        new_xyzp = knn_gather(xyzp, idx)[:, :, 0]

        grouped_xyz = knn_gather(xyz, idx)  # [B, out_points_num, k_nearest, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

        grouped_features = knn_gather(features, idx)  # [B, out_points_num, k_nearest, 3]
        new_features_concat = torch.cat([grouped_xyz_norm, grouped_features],
                                        dim=-1)  # [B, out_points_num, k_nearest, 3+F]
        new_features_concat = new_features_concat.permute(0, 3, 2, 1)  # [B, 3+F, k_nearest, out_points_num]
        new_features_concat = self.mlp(new_features_concat)  # [B, D', k_nearest, out_points_num]

        new_features_concat = torch.max(new_features_concat, 2)[0].transpose(1, 2)  # [B, out_points_num, D']
        return new_xyzp, new_features_concat


class TransConvBlock(nn.Module):
    def __init__(self, feature_num, hidden_num, k_nearest,  # Local Transformer args
                 global_point_num,  # Global Transformer args
                 conv, h, w) -> None:  # Conv args
        super().__init__()
        self.local_former = LocalTransformer(feature_num, hidden_num, k_nearest)
        if conv > 0:
            self.conv_former = ConvTransformer(feature_num, conv, k_nearest, h, w)
        if global_point_num > 0:
            self.global_former = GlobalTransfomer(feature_num, hidden_num, k_nearest, global_point_num)

    def forward(self, events, features):
        res = self.local_former(events, features)
        if hasattr(self, 'conv_former'):
            res = self.conv_former(events, res)
        if hasattr(self, 'global_former'):
            res = self.global_former(events, res)
        return res


class TransConvBackbone(nn.Module):
    def __init__(self, point_num, h, w, in_ch=4, hidden_ch=512, k_nearest=16, block_num=4,
                 global_ratios=[8, 4, 2, -1, -1], conv=[1, 1, 1, -1, -1], norm=True):
        super().__init__()
        self.block_num = block_num

        self.embedding = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        global_point_num = point_num // global_ratios[0] if global_ratios[0] > 0 else -1
        self.stem = TransConvBlock(feature_num=32, hidden_num=hidden_ch, k_nearest=k_nearest,
                                   global_point_num=global_point_num,
                                   conv=conv[0], h=h, w=w)

        self.down = nn.ModuleList()
        self.transformers = nn.ModuleList()
        in_ch, out_ch = 32, 32
        for i in range(block_num):
            h, w = h // 2, w // 2
            out_ch, point_num = out_ch * 2, point_num // 4
            self.down.append(TransitionDown(point_num, k_nearest, in_ch, out_ch, norm=norm))
            global_point_num = point_num // global_ratios[i + 1] if global_ratios[i + 1] > 0 else -1
            self.transformers.append(TransConvBlock(feature_num=out_ch, hidden_num=hidden_ch, k_nearest=k_nearest,
                                                    global_point_num=global_point_num,
                                                    conv=conv[i + 1], h=h, w=w))
            in_ch = out_ch

    def forward(self, events):
        features = self.stem(events, self.embedding(events))
        out = [(events, features)]

        for i in range(self.block_num):
            events, features = self.down[i](events, features)
            features = self.transformers[i](events, features)
            out.append((events, features))

        return out

#
# class TransConvHead(nn.Module):
#     def __init__(self, out_ch=32, hidden_ch=512, k_nearest=16, block_num=4):
#         super().__init__()
#         self.block_num = block_num
#
#         in_ch = 32 * (2 ** block_num)
#         self.connect_fc = nn.Sequential(
#             nn.Linear(in_ch, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, in_ch)
#         )
#         self.connect_transfomer = TransConvBlock(feature_num=in_ch, hidden_num=hidden_ch, k_nearest=k_nearest,
#                                                  global_point_num=-1, conv=-1, h=-1, w=-1)
#
#         self.up = nn.ModuleList()
#         self.transformers = nn.ModuleList()
#         down_ch = in_ch
#         for i in reversed(range(block_num)):
#             _out_ch = down_ch // 2 if i != 0 else out_ch
#             self.up.append(TransitionUp(down_ch, down_ch // 2, _out_ch))
#             self.transformers.append(TransConvBlock(feature_num=_out_ch, hidden_num=hidden_ch, k_nearest=k_nearest,
#                                                     global_point_num=-1, conv=-1, h=-1, w=-1))
#             down_ch = down_ch // 2
#
#     def forward(self, out_list):
#         xyz, features = out_list[-1]
#         features = self.connect_transfomer(xyz, self.connect_fc(features))
#
#         for i in range(self.block_num):
#             up_xyz, up_features = out_list[-i - 2]
#             up_features = self.up[i](xyz, features, up_xyz, up_features)
#             up_features = self.transformers[i](up_xyz, up_features)
#             xyz, features = up_xyz, up_features
#
#         return features


class TransConvCls(nn.Module):
    def __init__(self, cls_num, point_num, h, w, in_ch=4, hidden_ch=512, k_nearest=16, block_num=4,
                 global_ratios=[8, 4, 2, -1, -1], conv=[1, 1, 1, -1, -1]):
        super().__init__()
        self.backbone = TransConvBackbone(point_num, h, w, in_ch, hidden_ch, k_nearest, block_num, global_ratios, conv)
        if cls_num > 64:
            self.norm_pool = nn.Sequential(
                nn.LayerNorm(32 * (2 ** block_num)),
                SwapAxes(),
                nn.AdaptiveAvgPool1d(1),
                SwapAxes(),
            )
            self.classifier = nn.Linear(32 * (2 ** block_num), cls_num)
        else:
            _cls_num = 1 if cls_num == 2 else cls_num
            self.use_bce = True if cls_num == 2 else False
            self.classifier = nn.Sequential(
                nn.Linear(32 * (2 ** block_num), 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, _cls_num)
            )

    def forward(self, events):
        features = self.backbone(events)[-1][1]

        if hasattr(self, 'norm_pool'):
            features = self.norm_pool(features)  # B, 1, F

        result = self.classifier(features.mean(1))
        if self.use_bce:
            result = torch.sigmoid(result)
            if self.training:
                result = result.squeeze(1)
            else:
                result = result.repeat(1, 2)
                result[:, 0] = 1-result[:, 0]
        return result
        # features = torch.cat([features.mean(1), features.max(1)[0]], dim=-1)
        # return self.classifier(features)


# class TransConvSR(nn.Module):
#     def __init__(self, up_scale, point_num, h, w, in_ch=4, out_ch=64, hidden_ch=256, k_nearest=16, block_num=4,
#                  global_ratios=[8, 4, 2, -1, -1], conv=[1, 1, 1, -1, -1]):
#         super().__init__()
#         # assert up_scale in [2, 4, 16]
#         self.up_scale = up_scale
#         self.backbone = TransConvBackbone(point_num, h, w, in_ch, hidden_ch, k_nearest, block_num, global_ratios, conv,
#                                           norm=False)
#         self.head = TransConvHead(out_ch, hidden_ch, k_nearest, block_num)
#         self.out = TransConvBlock(out_ch, hidden_ch, k_nearest, -1, out_ch, 222, 124)
#         self.out_events = nn.Linear(out_ch, 3)
#
#     def forward(self, events):
#         out_list = self.backbone(events)
#         features = self.head(out_list)
#         B, N, F = features.shape
#
#         idx = knn_points(events[:, :, :3], events[:, :, :3], K=self.up_scale-1).idx
#         knn_events = knn_gather(events, idx)  # [B, N, up_scale-1, 4]
#         up_events = events[:, :, None]*2/3 + knn_events*1/3  # [B, N, up_scale-1, 4]
#         up_events = up_events.reshape([B, -1, 4])  # [B, N*(up_scale-1), 4]
#         up_events = torch.cat([events, up_events], dim=1).detach()  # [B, N*up_scale, 4]
#
#         # p = (events[..., 3][..., None] - 0.5) / 0.5  # [B, N_down, 1]
#         # pos = torch.clamp(p, 0, 1)
#         # neg = -torch.clamp(p, -1, 0)
#         # _features = torch.cat([pos, neg, features], dim=-1)  # [B, N, F+2]
#         _features = features
#         knn_features = knn_gather(_features, idx)  # [B, N, up_scale-1, F+2]
#         up_features = _features[:, :, None]*2/3. + knn_features*1/3.  # [B, N*(up_scale-1), F+2]
#         up_features = up_features.reshape([B, -1, F])  # [B, N*(up_scale-1), F+2]
#         up_features = torch.cat([_features, up_features], dim=1)  # [B, N*up_scale, 4]
#
#         up_features = self.out(up_events, up_features)
#         bias = self.out_events(up_features)
#         # up_events[:, :, 3] = torch.sigmoid(up_events[:, :, 3])
#         if self.training:
#             return up_events[:, :, :3], bias
#         else:
#             up_events[:, :, :3] = up_events[:, :, 3] + bias
#             return up_events


class SRTransformer(nn.Module):
    def __init__(self, feature_num, hidden_num, k_nearest) -> None:
        super().__init__()
        '''
        According to Point Transfomer Sec 3.2
        '''
        self.k_nearest = k_nearest

        self.kernel_fn = nn.Linear(feature_num, hidden_num)
        self.aggregation_fn = nn.Linear(hidden_num, feature_num)

        self.w_qs = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_ks = nn.Linear(hidden_num, hidden_num, bias=False)
        self.w_vs = nn.Linear(hidden_num, hidden_num, bias=False)

        self.pos_encoder = nn.Sequential(
            nn.Linear(4+4+4+4, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyzp, features):
        xyz = xyzp[:, :, :3]
        knn_result = knn_points(xyz, xyz, K=self.k_nearest)
        idx = knn_result.idx
        pos = xyzp[:, :, None].repeat(1, 1, self.k_nearest, 1) # [B, N, k_nearest, 4]
        pos_k = knn_gather(xyzp, idx) # [B, N, k_nearest, 4]
        pos = torch.cat([pos, pos_k, pos-pos_k, torch.sqrt(torch.pow((pos-pos_k),2))], dim=-1)
        pos_enc = self.pos_encoder(pos)  # [B, N, k_nearest, hidden_num]

        x = self.kernel_fn(features)
        # q: [B, N, hidden_num], k: [B, N, k_nearest, hidden_num], v: [B, N, k_nearest, hidden_num]
        q, k, v = self.w_qs(x), knn_gather(self.w_ks(x), idx), knn_gather(self.w_vs(x), idx)
        attention = self.attention_fn(q[:, :, None] - k + pos_enc)  # [B, N, k_nearest, hidden_num]
        attention = F.softmax(attention / np.sqrt(k.size(-1)), dim=-2)  # [B, N, k_nearest, hidden_num]

        res = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)
        res = self.aggregation_fn(res) + features
        # return res, attention
        return res


class TransConvSR(nn.Module):
    def __init__(self, up_scale, point_num, h, w, in_ch=4, out_ch=64, hidden_ch=256, k_nearest=16, block_num=4,
                 global_ratios=[8, 4, 2, -1, -1], conv=[1, 1, 1, -1, -1]):
        super().__init__()
        # assert up_scale in [2, 4, 16]
        self.up_scale = up_scale
        self.embedding = nn.Sequential(
            nn.Linear(in_ch, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.transformer = SRTransformer(feature_num=512, hidden_num=512, k_nearest=k_nearest)
        self.output = nn.Linear(512, 3)

    def forward(self, events):
        features = self.embedding(events)

        B, N, F = features.shape
        idx = knn_points(events[:, :, :3], events[:, :, :3], K=self.up_scale-1).idx
        knn_events = knn_gather(events, idx)  # [B, N, up_scale-1, 4]
        up_events = events[:, :, None]*2/3 + knn_events*1/3  # [B, N, up_scale-1, 4]
        up_events = up_events.reshape([B, -1, 4])  # [B, N*(up_scale-1), 4]
        up_events = torch.cat([events, up_events], dim=1).detach()  # [B, N*up_scale, 4]

        knn_features = knn_gather(features, idx)  # [B, N, up_scale-1, F+2]
        up_features = features[:, :, None]*2/3. + knn_features*1/3.  # [B, N*(up_scale-1), F+2]
        up_features = up_features.reshape([B, -1, F])  # [B, N*(up_scale-1), F+2]
        up_features = torch.cat([features, up_features], dim=1)  # [B, N*up_scale, 4]

        up_features = self.transformer(up_events, up_features)
        bias = self.output(up_features)

        if self.training:
            return up_events[:, :, :3].detach(), bias
        else:
            up_events[:, :, :3] = up_events[:, :, 3] + bias
            return up_events
