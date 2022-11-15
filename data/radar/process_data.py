import os
import sys
import numpy as np
import csv
import json
from sklearn.cluster import DBSCAN

def process_radar_data(root, fps=16, train=True, eps=0.05, min_samples=3, nradars=None):

    point_clouds = {}

    with open(os.path.join(root, 'metadata.json'), 'r') as metadata:
            offset_map = json.load(metadata)

    for file_name in os.listdir(root):
            if file_name.endswith(".csv"):
                name_parts = file_name.split('.')[0].split('_')
                if name_parts[0] == "label":
                    with open(os.path.join(root, file_name)) as csv_file:
                        video = np.array(list(csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
                        video[:,1] -= np.amin(video[:,1])

                    if (name_parts[1], name_parts[2][:2]) not in point_clouds.keys():
                        if nradars is None:
                            if train:
                                point_clouds[(name_parts[1], name_parts[2][:2])] = [None] * 4
                            else:
                                point_clouds[(name_parts[1], name_parts[2][:2])] = [None] * 2
                        else:
                            point_clouds[(name_parts[1], name_parts[2][:2])] = [None] * nradars

                    for key in offset_map.keys():    
                        if key in name_parts[2] and ord(key) - ord('a') < len(point_clouds[(name_parts[1], name_parts[2][:2])]):
                            offsets = offset_map[key]
                            offset_rot = offsets[3]
                            offset_trans = offsets[:3]
                            video[:,3:6] = np.matmul(video[:,3:6], np.array([[np.cos(offset_rot), np.sin(offset_rot), 0], 
                                                    [- np.sin(offset_rot), np.cos(offset_rot), 0], 
                                                    [0, 0, 1]])) + np.array(offset_trans)
                            point_clouds[(name_parts[1], name_parts[2][:2])][ord(key) - ord('a')] = video
                            print("Detected key", key, "and giving offsets", offsets)
                            break

    comb_point_clouds = {}
    new_point_clouds = {}

    frames_skipped = 0

    for key in point_clouds.keys():
        comb_point_clouds[key] = np.concatenate(point_clouds[key], axis=0)
        print("Initial number of points:", comb_point_clouds[key].shape[0])

        max_time = np.amax(comb_point_clouds[key][:,0])
        frame_period = 1 / fps
        n_frames = int(max_time // frame_period + 1)
        print("Number of frames:", n_frames)

        new_point_clouds[key] = []
        for frame_n in range(n_frames):
            frame_mask = np.logical_and(frame_n * frame_period <= comb_point_clouds[key][:,0], comb_point_clouds[key][:,0] < (frame_n + 1) * frame_period)
            if comb_point_clouds[key][frame_mask, :].shape[0] == 0:
                print("Zero point frames detected from time", frame_n * frame_period, "to", (frame_n + 1) * frame_period)
                frames_skipped = frames_skipped + 1
                continue
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(comb_point_clouds[key][frame_mask, 3:-1])
            filtered_points = comb_point_clouds[key][frame_mask,2:][dbscan.labels_ != -1,:]
            frame_indicator = np.ones((filtered_points.shape[0], 1)) * (frame_n - frames_skipped)
            new_point_clouds[key].append(np.concatenate((frame_indicator, filtered_points), axis=1))
        
        new_point_clouds[key] = np.concatenate(new_point_clouds[key], axis=0)

        print("New number of points:", new_point_clouds[key].shape[0])

    if train:
        new_root = os.path.join(root, "../train")
    else:
        new_root = os.path.join(root, "../test")

    if not os.path.exists(new_root):
        os.makedirs(new_root)

    for key in new_point_clouds.keys():
        new_file_name = "label_" + key[0] + "_" + key[1] + ".npy"
        with open(os.path.join(new_root, new_file_name), 'wb') as new_file:
            np.save(new_file, new_point_clouds[key])

if __name__ == "__main__":
    process_radar_data(root='/workspace/P4Transformer/data/radar/set6', eps=0.04, min_samples=3, nradars=3)
    process_radar_data(root='/workspace/P4Transformer/data/radar/set6', eps=0.06, min_samples=5, train=False, nradars=1)
    