import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points, ball_query


class TransformerBlock(nn.Module):
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
            nn.Linear(3, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )
        self.attention_fn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num)
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        knn_result = knn_points(xyz, xyz, K=self.k_nearest)
        dists, idx = knn_result.dists, knn_result.idx
        knn_xyz = knn_gather(xyz, idx)  # [B, N, k_nearest, 3]
        pos_enc = self.pos_encoder(xyz[:, :, None] - knn_xyz)  # [B, N, k_nearest, hidden_num]

        dense = features
        x = self.kernel_fn(features)
        # q: [B, N, hidden_num], k: [B, N, k_nearest, hidden_num], v: [B, N, k_nearest, hidden_num]
        q, k, v = self.w_qs(x), knn_gather(self.w_ks(x), idx), knn_gather(self.w_vs(x), idx)

        attention = self.attention_fn(q[:, :, None] - k + pos_enc)  # [B, N, k_nearest, hidden_num]
        attention = F.softmax(attention / np.sqrt(k.size(-1)), dim=-2)  # [B, N, k_nearest, hidden_num]

        res = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)
        res = self.aggregation_fn(res) + dense
        return res, attention


class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class TransitionDown(nn.Module):
    def __init__(self, out_point_num, k_nearest, in_ch, out_ch, hidden_layer_num=2):
        super().__init__()
        self.out_point_num = out_point_num
        self.k_nearest = k_nearest
        self.mlp = nn.Sequential()
        for i in range(hidden_layer_num):
            _in_ch = in_ch + 3 if i == 0 else out_ch
            self.mlp.add_module('mlp_conv_{}'.format(i), nn.Conv2d(_in_ch, out_ch, 1))
            self.mlp.add_module('mlp_bn_{}'.format(i), nn.BatchNorm2d(out_ch))
            self.mlp.add_module('relu_{}'.format(i), nn.ReLU())

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, F]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_features_concat: sample points feature data, [B, S, D']
        """

        new_xyz, _ = sample_farthest_points(xyz, K=self.out_point_num)  # [B, out_points_num, 3]
        knn_result = knn_points(new_xyz, xyz, K=self.k_nearest)
        dists, idx = knn_result.dists, knn_result.idx

        grouped_xyz = knn_gather(xyz, idx)  # [B, out_points_num, k_nearest, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

        grouped_features = knn_gather(features, idx)  # [B, out_points_num, k_nearest, 3]
        new_features_concat = torch.cat([grouped_xyz_norm, grouped_features],
                                        dim=-1)  # [B, out_points_num, k_nearest, 3+F]
        new_features_concat = new_features_concat.permute(0, 3, 2, 1)  # [B, 3+F, k_nearest, out_points_num]
        new_features_concat = self.mlp(new_features_concat)  # [B, D', k_nearest, out_points_num]

        new_features_concat = torch.max(new_features_concat, 2)[0].transpose(1, 2)  # [B, out_points_num, D']
        return new_xyz, new_features_concat


class TransfomerBackbone(nn.Module):
    def __init__(self, point_num, in_ch=4, hidden_ch=512, k_nearest=16, block_num=4):
        super().__init__()
        self.block_num = block_num

        self.kernel_fn = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.kernel_transformer = TransformerBlock(32, hidden_ch, k_nearest)

        self.down = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.transformers2 = nn.ModuleList()
        in_ch, out_ch = 32, 32
        for i in range(block_num):
            out_ch, point_num = out_ch * 2, point_num // 4
            self.down.append(TransitionDown(point_num, k_nearest, in_ch, out_ch))
            self.transformers.append(TransformerBlock(out_ch, hidden_ch, k_nearest))
            if i < 3:
                self.transformers2.append(TransformerBlock(out_ch, hidden_ch, k_nearest))
            in_ch = out_ch

    def forward(self, events):
        xyz = events[..., :3]
        features, _ = self.kernel_transformer(xyz, self.kernel_fn(events))
        out = [(xyz, features)]

        for i in range(self.block_num):
            xyz, features = self.down[i](xyz, features)
            features, _ = self.transformers[i](xyz, features)
            if i < 3:
                features, _ = self.transformers2[i](xyz, features)
            out.append((xyz, features))

        return out

class TransCls(nn.Module):
    def __init__(self, cls_num, point_num, in_ch=4, hidden_ch=512, k_nearest=16, block_num=4):
        super().__init__()
        self.backbone = TransfomerBackbone(point_num, in_ch, hidden_ch, k_nearest, block_num)
        hidden_ch = 64 if cls_num < 64 else 128
        self.classifier = nn.Sequential(
            nn.Linear(32 * (2 ** block_num), 256),
            nn.ReLU(),
            nn.Linear(256, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, cls_num)
        )

    def forward(self, events):
        features = self.backbone(events)[-1][1]
        return self.classifier(features.mean(1))

