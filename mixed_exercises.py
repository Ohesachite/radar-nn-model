import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from scipy.signal import correlate
from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
import csv
import time
import json
import importlib

segmod = importlib.import_module("get-count-accs")

import torch
from torch import nn
import torch.nn.functional as F
import models.radar as Models

# Segmentation

file_name1 = 'data/radar/set8_120/label_ln_02a.csv'
file_name2 = 'data/radar/set9_120/label_sq_02a.csv'

point_cloud_vid_p1 = segmod.load_point_cloud_vid([file_name1], [[0.0, 0.0, 0.0, 0.0]])
point_cloud_vid_p2 = segmod.load_point_cloud_vid([file_name2], [[0.0, 0.0, 0.0, 0.0]])

point_cloud_vid = point_cloud_vid_p1 + point_cloud_vid_p2
    
start_time = time.time()
vid_len = len(point_cloud_vid)
new_point_cloud_vid, median_point = segmod.median_centering(point_cloud_vid)
fft_size = 64
dim_min, dim_max = -1.2, 1.2 - 2.4/fft_size
xi, yi, zi = np.meshgrid(np.linspace(dim_min, dim_max, fft_size), np.linspace(dim_min, dim_max, fft_size), np.linspace(dim_min, dim_max, fft_size))
doppler_vals_over_time = []
spatial_vals_over_time = []
for pc in new_point_cloud_vid:
    doppler_vals = griddata(pc[:,2:5], pc[:,5], (xi, yi, zi), method='linear', fill_value=0.0)
    # spatial_vals = griddata(pc[:,2:5], np.ones(pc.shape[0]), (xi, yi, zi), method='linear', fill_value=0.0)
    doppler_vals_over_time.append(doppler_vals)

doppler_vals_over_time = np.array(doppler_vals_over_time)
end_time = time.time()
print("Interpolation step done in", end_time - start_time, "seconds")

window_size = 24
template_found = False
initial_template_size = 64
template_alpha = 0.5

segment_index = 0
max_score = - np.inf
max_score_props = (-1, -1)
count = 0

scores = []

start_time = time.time()
ssm_save = np.zeros((vid_len, vid_len))
dv_txy = np.mean(doppler_vals_over_time, axis=3)
dv_txz = np.mean(doppler_vals_over_time, axis=2)
dv_tyz = np.mean(doppler_vals_over_time, axis=1)
for i in range(vid_len):
    for j in range(vid_len):
        xy_correlation = np.mean(np.multiply(dv_txy[i,:,:], dv_txy[j,:,:]))
        xz_correlation = np.mean(np.multiply(dv_txz[i,:,:], dv_txz[j,:,:]))
        yz_correlation = np.mean(np.multiply(dv_tyz[i,:,:], dv_tyz[j,:,:]))

        ssm_save[i,j] = xy_correlation + xz_correlation + yz_correlation

end_time = time.time()
print("Correlation took", end_time-start_time, "seconds")

start_time = time.time()

bounds = segmod.find_boundaries(ssm_save)
bounds = segmod.dynamic_bound_combination(bounds, 'n/a')

end_time = time.time()
print("Segmentation took", end_time-start_time, "seconds")

# Data preparation

prepared_video = [0] * vid_len

for i, p in enumerate(point_cloud_vid):
    if p.shape[0] > 1024:
        r = np.random.choice(p.shape[0], size=1024, replace=False)
    else:
        repeat, residue = 1024 // p.shape[0], 1024 % p.shape[0]
        r = np.random.choice(p.shape[0], size=residue, replace=False)
        r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
    prepared_video[i] = p[r, :]

prepared_video = np.array(prepared_video)

prepared_features = prepared_video[:,:,5:].astype(np.float32)
prepared_video = prepared_video[:,:,2:5].astype(np.float32)

print(prepared_video.shape, prepared_features.shape)

print("Initializing torch")
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda')

print("Creating model")
model = Models.RadarP4Transformer(radius=0.5, nsamples=32, spatial_stride=32,
                                temporal_kernel_size=3, temporal_stride=2,
                                emb_relu=False,
                                dim=1024, depth=5, heads=8, dim_head=128,
                                mlp_dim=2048, num_classes=10)

print("Loading checkpoint")
checkpoint = torch.load('ckpts/ckpt_39.pth')
model.load_state_dict(checkpoint['model'])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

model.eval()
print("Generating clips")
with torch.no_grad():
    for bound_l, bound_r in bounds:
        total_prob = np.zeros((1, 10))
        if bound_r - bound_l < 23:
            clip = torch.unsqueeze(torch.tensor(prepared_video[bound_l:bound_l+24,:,:], device=device), 0)
            features = torch.unsqueeze(torch.tensor(np.swapaxes(prepared_features[bound_l:bound_l+24,:,:], 1, 2), device=device), 0)
            
            output, _, _ = model(clip, features)
            prob = F.softmax(output, dim=1)

            prob = prob.cpu().numpy()
            total_prob += prob
        else:
            for t in range(bound_l, bound_r-22):
                clip = torch.unsqueeze(torch.tensor(prepared_video[t:t+24,:,:], device=device), 0)
                features = torch.unsqueeze(torch.tensor(np.swapaxes(prepared_features[t:t+24,:,:], 1, 2), device=device), 0)
                
                output, _, _ = model(clip, features)
                prob = F.softmax(output, dim=1)

                prob = prob.cpu().numpy()
                total_prob += prob
        pred = np.argmax(total_prob)
        print(pred)