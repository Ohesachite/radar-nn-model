import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from scipy.signal import correlate
from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits import mplot3d
import csv
import time
import json
import importlib

segmod = importlib.import_module("get-count-accs")

import torch
from torch import nn
import torch.nn.functional as F
import models.radar as Models

# Files (You may need to change file path if the data you have doesn't exist, or try finding the data on OneDrive)

# file_name2 = 'data/radar/set9_120/label_sq_02c.csv'
file_name2 = 'data/radar/setenv-457_unseentar/label_su_01c.csv'

# Preprocessing

# point_cloud_vid_p1 = segmod.load_point_cloud_vid([file_name1], [[0.0, 0.0, 0.0, 0.0]])
point_cloud_vid_p2, skipped_indices = segmod.load_point_cloud_vid([file_name2], [[0.0, 0.0, 0.0, 0.0]], retskipped=True)
n_skipped = len(skipped_indices)

point_cloud_vid = point_cloud_vid_p2

# Segmentation
    
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

# Data preparation for classification

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
checkpoint = torch.load('ckpts/sw-3seen-new/alt_ckpt_39.pth')
model.load_state_dict(checkpoint['model'])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Classification (sliding window within segments)

model.eval()
print("Generating clips")
preds = []
with torch.no_grad():
    for bound_l, bound_r in bounds:
        total_prob = np.zeros((1, 10))
        print("Bounds", bound_l, "to", bound_r)
        if bound_r - bound_l < 23:
            clip = torch.unsqueeze(torch.tensor(prepared_video[bound_l:bound_l+24,:,:], device=device), 0)
            features = torch.unsqueeze(torch.tensor(np.swapaxes(prepared_features[bound_l:bound_l+24,:,:], 1, 2), device=device), 0)
            
            output, _, _ = model(clip, features)
            prob = F.softmax(output, dim=1)

            prob = prob.cpu().numpy()
            total_prob += prob
            print(np.argmax(prob))
        else:
            for t in range(bound_l, bound_r-22):
                clip = torch.unsqueeze(torch.tensor(prepared_video[t:t+24,:,:], device=device), 0)
                features = torch.unsqueeze(torch.tensor(np.swapaxes(prepared_features[t:t+24,:,:], 1, 2), device=device), 0)
                
                output, _, _ = model(clip, features)
                prob = F.softmax(output, dim=1)

                prob = prob.cpu().numpy()
                total_prob += prob
                print(np.argmax(prob))
        pred = np.argmax(total_prob)
        preds.append(pred)

print("Predictions:", preds)

# Animation

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
plot = ax.scatter3D([], [], [], animated=True)
txt = fig.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
radarLoc = ax.scatter3D(0.0, 0.0, 0.0, animated=True)
radarLocLabel = ax.text(0.0, 0.0, 0.0, "Radar Location")
plot.set_cmap("viridis")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-3, 3)
ax.set_ylim(0, 5)
ax.set_zlim(-1.2, 1.5)
ax.view_init(azim=0, elev=90)

exercises = ['Idle', 'Pectoral Fly', 'Arms Up Down', 'Legs Up Down', 'Squat', 'Lunge', 'Push Up', 'Jumping Jacks', 'Torso Rotation', 'Sit Ups']

def init():
    all_plots = []
    
    plot._offsets3d = ([], [], [])
    txt.set_text('No exercises done!')
    all_plots.append(plot)
        
    return all_plots

cnts = {}

def animate(i):
    all_plots = []
    vi = i

    for si in skipped_indices:
        if si == vi:
            plot._offsets3d = ([], [], [])
            plot.set_array([])
            all_plots.append(plot)

            return all_plots

        elif si < vi:
            vi -= 1
    
    inc_indices = [bound[1] for bound in bounds]
    if vi in inc_indices:
        idx = sum([1 for j in inc_indices if vi >= j])-1
        print(vi, idx)
        exercise = exercises[preds[idx]]
        if exercise in cnts.keys():
            cnts[exercise] += 1
        else:
            cnts[exercise] = 1

        cnt_str = 'RF-HAC Output: ' + ', '.join([ex + ': ' + str(cnt) for ex, cnt in cnts.items()])
        txt.set_text(cnt_str)
            
    plot_points = point_cloud_vid[vi]
    plot._offsets3d = (plot_points[:,2], plot_points[:,3], plot_points[:,4])
    plot_weights = (plot_points[:,5] + 3) / 6
    plot.set_array(plot_weights)
    all_plots.append(plot)
    
    return all_plots

animation = anim.FuncAnimation(fig, func=animate, frames=vid_len+n_skipped, init_func=init, interval=60)
animation.save('results/' + 'pc2.mp4', writer='ffmpeg', fps=16)

print(cnts)
