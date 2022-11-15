import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer import *

class RadarP4Transformer (nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                    # P4DConv: spatial
                temporal_kernel_size, temporal_stride,                                      # P4DConv: temporal
                emb_relu,                                                                   # embedding: relu
                dim, depth, heads, dim_head,                                                # transformer
                mlp_dim, num_classes):                                                      # output
        super().__init__()

        # self.conv1 = P4DConv(in_planes=2, 
        #                     mlp_planes=[32, 64],
        #                     mlp_batch_norm=[True, True],
        #                     mlp_activation=[True, True],
        #                     spatial_kernel_size=[radius, nsamples],
        #                     spatial_stride=4,
        #                     temporal_kernel_size=1,
        #                     temporal_stride=1,
        #                     temporal_padding=[0,0],
        #                     operator='*')
        
        self.conv2 = P4DConv(in_planes=2, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='*', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.constrastive_mlp = nn.Linear(4, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, xyzs, old_features):                                                                                              # [B, L, N, 3], [B, L, 1, n]
        device = xyzs.get_device()
        # xyzs_s1, features_s1 = self.conv1(xyzs, old_features)
        new_xyzs, features = self.conv2(xyzs, old_features)                                                                             # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        new_xyzs = torch.split(tensor=new_xyzs, split_size_or_sections=1, dim=1)
        new_xyzs = [torch.squeeze(xyz, dim=1).contiguous() for xyz in new_xyzs]
        for t, xyz in enumerate(new_xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(xyzts, (xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                                   # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                         # [B, L,   n, C]
        features = torch.reshape(features, (features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))                 # [B, L*n, C] 

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        contrastive_xyzts = self.constrastive_mlp(xyzts)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output, contrastive_xyzts, features