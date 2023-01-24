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
        
        self.tube_embedding = P4DConv(in_planes=2, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='*', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, xyzs, old_features):                                                                                              # [B, L, N, 3], [B, L, 1, n]
        device = xyzs.get_device()
        new_xyzs, features = self.tube_embedding(xyzs, old_features)                                                                    # [B, L, n, 3], [B, L, C, n] 

        xyzts = self.xyzs_to_xyzts(new_xyzs, device)

        features = features.permute(0, 1, 3, 2)                                                                                         # [B, L,   n, C]
        features = torch.reshape(features, (features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))                 # [B, L*n, C] 

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        new_features = self.transformer(embedding)
        output = torch.max(new_features, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output, xyzts, features

    def xyzs_to_xyzts(self, xyzs, device):
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(xyzts, (xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                                   # [B, L*n, 4]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        return xyzts

class PointCloudEncoder (nn.Module):
    def __init__(self, radius, nsamples, embedding_dim=1024):

        # Spatial Convolution
        self.conv1 = P4DConv(in_planes=2,
                             mlp_planes=[32,64,128],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=8,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        # Temporal Convolution
        self.conv2 = P4DConv(in_planes=128,
                             mlp_planes=[256, 512, embedding_dim],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*radius, nsamples],
                             temporal_kernel_size=3,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

    def forward(xyzs, features):

        xyzs, features = self.conv1(xyzs, features)
        xyzs, features = self.conv2(xyzs, features)

        embeddings = torch.max(features, dim=2, keepdim=False)[0]

        return xyzs, embeddings

class RadarPeriodEstimator (nn.Module):
    def __init__(self, radius, nsamples, 
                embedding_dim=1024)

        self.encoder = PointCloudEncoder(radius, nsamples, embedding_dim=embedding_dim)

        self.conv_3x3_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
            nn.ReLu()
        )

    def forward(xyzs, features):

        xyzs, embeddings = self.encoder(xyzs, features)

        features = self.get_tsm(embeddings)

        features = self.conv_3x3_layer(features)
        features = torch.reshape(features, (features.shape[0], features.shape[1], -1))

    def get_tsm(embeddings, temperature=13.544):
        
        sims = torch.cdist(embeddings, embeddings, p=2.0)
        sims /= temperature
        sims = F.softmax(sims, dim=-1)
        sims = torch.unsqueeze(sims, -1)