import os
import sys
import numpy as np
import csv
import json
from sklearn.cluster import DBSCAN

def process_radar_data(root, fps=16, train=True, eps=0.05, min_samples=3, radars=None, label_offset=None):

    point_clouds = {}
    count_data = {}

    with open(os.path.join(root, 'metadata.json'), 'r') as metadata:
        offset_map = json.load(metadata)

    for file_name in os.listdir(root):
        if file_name == "cycle_data.csv":
            with open(os.path.join(root, file_name)) as cycle_file:
                reader = csv.reader(cycle_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
                read_data = []
                cycle_data = {}
                for np1, np2, sta, end, cnt in reader:
                    values = [str(np1), str(np2), int(sta), int(end), int(cnt)]
                    read_data.append(values)
                for values in read_data:
                    cycle_data[(values[0], values[1])] = [values[2], values[3], values[4]]
        elif file_name.endswith(".csv"):
            name_parts = file_name.split('.')[0].split('_')
            if name_parts[0] == "label":
                with open(os.path.join(root, file_name)) as csv_file:
                    video = np.array(list(csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
                    video[:,7] /= 10000
                    video[:,6] /= 2

                if (name_parts[1], name_parts[2][:2]) not in point_clouds.keys():
                    point_clouds[(name_parts[1], name_parts[2][:2])] = []

                for key in offset_map.keys():    
                    if key in name_parts[2] and (ord(key) - ord('a')) in radars:
                        if len(radars) > 1:
                            offsets = offset_map[key]
                            offset_rot = offsets[3]
                            offset_trans = offsets[:3]
                            video[:,3:6] = np.matmul(video[:,3:6], np.array([[np.cos(offset_rot), np.sin(offset_rot), 0], 
                                                    [- np.sin(offset_rot), np.cos(offset_rot), 0], 
                                                    [0, 0, 1]])) + np.array(offset_trans)
                            print("Detected key", key, "and giving offsets", offsets)
                        else:
                            print("Detected key", key, "but not offseting")
                        point_clouds[(name_parts[1], name_parts[2][:2])].append(video)
                        break

    comb_point_clouds = {}
    new_point_clouds = {}

    for key in point_clouds.keys():
        comb_point_clouds[key] = np.concatenate(point_clouds[key], axis=0)
        print("Initial number of points:", comb_point_clouds[key].shape[0])

        max_time = np.amax(comb_point_clouds[key][:,0])
        frame_period = 1 / fps
        n_frames = int(max_time // frame_period + 1)
        print("Number of frames:", n_frames)

        frames_skipped = 0

        new_point_clouds[key] = []
        for frame_n in range(n_frames):
            frame_mask = np.logical_and(frame_n * frame_period <= comb_point_clouds[key][:,0], comb_point_clouds[key][:,0] < (frame_n + 1) * frame_period)
            if comb_point_clouds[key][frame_mask, :].shape[0] == 0:
                print("Zero point frames detected from time", frame_n * frame_period, "to", (frame_n + 1) * frame_period)
                frames_skipped = frames_skipped + 1
                if key in cycle_data.keys():
                    if frame_n < cycle_data[key][0]:
                        cycle_data[key][0] = cycle_data[key][0] - 1
                    elif frame_n >= n_frames - cycle_data[key][1]:
                        cycle_data[key][1] = cycle_data[key][1] - 1
                continue
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(comb_point_clouds[key][frame_mask, 3:6])
            if all(dbscan.labels_ == -1):
                print("DBSCAN caused zero point frame from time", frame_n * frame_period, "to", (frame_n + 1) * frame_period, "! Dropping the frame!")
                frames_skipped = frames_skipped + 1
                if key in cycle_data.keys():
                    if frame_n < cycle_data[key][0]:
                        cycle_data[key][0] = cycle_data[key][0] - 1
                    elif frame_n >= n_frames - cycle_data[key][1]:
                        cycle_data[key][1] = cycle_data[key][1] - 1
                continue
            filtered_points = comb_point_clouds[key][frame_mask,2:][dbscan.labels_ != -1,:]
            frame_indicator = np.ones((filtered_points.shape[0], 1)) * (frame_n - frames_skipped)
            new_point_clouds[key].append(np.concatenate((frame_indicator, filtered_points), axis=1))
        
        new_point_clouds[key] = np.concatenate(new_point_clouds[key], axis=0)

        print("New number of points:", new_point_clouds[key].shape[0])
        if key in cycle_data.keys():
            print("Start frame:", cycle_data[key][0], ", End frame:", n_frames - frames_skipped - cycle_data[key][1] - 1, ", Count:", cycle_data[key][2])

    if train:
        new_root = os.path.join(root, "../train")
    else:
        new_root = os.path.join(root, "../test")

    if not os.path.exists(new_root):
        os.makedirs(new_root)

    for key in new_point_clouds.keys():
        point_cloud_and_metadata = {}

        if label_offset is not None:
            if key[1][0] == '0':
                label_num = int(key[1][1]) + label_offset
            else:
                label_num = int(key[1]) + label_offset
            
            if label_num < 10:
                new_label = '0' + str(label_num)
            else:
                new_label = str(label_num)

            new_file_name = "label_" + key[0] + "_" + new_label + ".npz"
        else:
            new_file_name = "label_" + key[0] + "_" + key[1] + ".npz"

        with open(os.path.join(new_root, new_file_name), 'wb') as new_file:
            point_cloud_and_metadata["Point cloud"] = new_point_clouds[key]
            if key in cycle_data.keys():
                point_cloud_and_metadata["Cycle data"] = cycle_data[key]
            else:
                print("No cycle data for", key)
            np.save(new_file, point_cloud_and_metadata)

if __name__ == "__main__":
    base_folder = '/home/alan/Documents/radar-nn-model/data/radar'
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.04, min_samples=3, radars=[0,1,2])                             # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=3)            # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=6)           # 7-9
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=9)           # 10-12
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=12)          # 13-15
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=15)          # 16-18
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[0], label_offset=18)                # 19-21
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[1], label_offset=21)                # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[2], label_offset=24)                # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[0])                   # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=3)   # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=6)   # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=9)  # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=12) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=15) # 34-36
    process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=18) # 19-21
    process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=21) # 22-24
    process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=24) # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=27) # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=30) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=33) # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=36) # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=39) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=42) # 34-36