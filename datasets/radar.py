import os
import sys
import numpy as np
import random
from torch.utils.data import Dataset
import csv
import json

class Radar(Dataset):
    def __init__(self, root, frames_per_clip=32, frame_interval=1, num_points=2048, mask_split=[1.0, 0.0, 0.0]):
        super(Radar, self).__init__()

        self.videos = []
        self.generate_positive = {}
        self.labels = []
        index = 0
        point_clouds = {}
        label_key = {
            "idle": 0,
            "pf": 1,
            "aud": 2,
            "lud": 3,
            "sq": 4,
            "ln": 5,
            "pu": 6,
            "jj": 7,
            "tr": 8,
            "su": 9
        }

        self.index_map = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        assert(sum(mask_split) == 1.0)

        positive_file = False
        if os.path.exists(os.path.join(root, "vid_metadata.json")):
            metadata_file = open(os.path.join(root, "vid_metadata.json"))
            metadata = json.load(metadata_file)
            metadata_file.close()
            positive_file = True

        vid_index_num_map = {}
        vid_num_index_map = {}

        idx = 0
        for file_name in os.listdir(root):
            if file_name.endswith(".npy"):
                name_parts = file_name.split('.')[0].split('_')
                if name_parts[0] == "label":
                    point_clouds[(name_parts[1], name_parts[2])] = np.load(os.path.join(root, file_name))
                    self.videos.append(point_clouds[(name_parts[1], name_parts[2])])
                    self.labels.append(label_key[name_parts[1]])

                    vid_index_num_map[(index, label_key[name_parts[1]])] = name_parts[2]
                    vid_num_index_map[(name_parts[2], label_key[name_parts[1]])] = index

                    nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                    for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                        self.index_map.append((index, t))
                        if t < (nframes-frame_interval*(frames_per_clip-1)) * mask_split[0]:
                            self.train_indices.append(idx)
                        elif t < (nframes-frame_interval*(frames_per_clip-1)) * (mask_split[0] + mask_split[1]):
                            self.val_indices.append(idx)
                        else:
                            self.test_indices.append(idx)
                        idx += 1
                    index += 1

        for key, value in vid_index_num_map.items():
            if not positive_file:
                self.generate_positive[key[0]] = [-1]
            elif positive_file and len(metadata[value]) == 0:
                self.generate_positive[key[0]] = [-1]
            else:
                self.generate_positive[key[0]] = []
                for num in metadata[value]:
                    positive_index = vid_num_index_map[(num, key[1])]
                    self.generate_positive[key[0]].append(positive_index)

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.num_classes = len(label_key)

    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]
 
        clip = []
        for i in range(self.frames_per_clip):
            frame = video[video[:,0] == t+i*self.frame_interval, 2:]
            clip.append(frame)

        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]

        clip = np.array(clip)

        clip_points = clip[:,:,:3]
        clip_features = clip[:,:,3:]
        clip_features = np.swapaxes(clip_features, 1, 2)

        positive_clip_points, positive_clip_features = self.generate_positive_sample(clip_points, clip_features, idx)

        return clip_points.astype(np.float32), clip_features.astype(np.float32), label, index, positive_clip_points.astype(np.float32), positive_clip_features.astype(np.float32)

    def generate_positive_sample(self, clip_points, clip_features, idx):
        if idx not in self.train_indices:
            return clip_points, clip_features

        index, _ = self.index_map[idx]

        if self.generate_positive[index][0] != -1:
            index = random.choice(self.generate_positive[index])
            nframes = int(np.amax(self.videos[index][:,0])) + 1
            t = random.choice(range(0, nframes-self.frame_interval*(self.frames_per_clip-1)))

            video = self.videos[index]
            label = self.labels[index]
    
            new_clip = []
            for i in range(self.frames_per_clip):
                frame = video[video[:,0] == t+i*self.frame_interval, 2:]
                new_clip.append(frame)

            for i, p in enumerate(new_clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                new_clip[i] = p[r, :]

            new_clip = np.array(new_clip)

            new_clip_points = new_clip[:,:,:3]
            new_clip_features = new_clip[:,:,3:]
            new_clip_features = np.swapaxes(new_clip_features, 1, 2)

        else:
            random_sign = np.random.randint(2) * 2 - 1
            random_angular_offset = np.random.randint(180) + 1
            generated_angle_deg = random_sign * random_angular_offset
            generated_angle_rad = generated_angle_deg * np.pi / 180

            aggregated_clip_points = np.reshape(clip_points, (self.frames_per_clip * self.num_points, 3))
            median_point = np.median(aggregated_clip_points, axis=0)

            new_clip_points = []

            for i, p in enumerate(clip_points):
                p = p - median_point
                p = np.matmul(p, np.array([[np.cos(generated_angle_rad), np.sin(generated_angle_rad), 0], 
                                            [- np.sin(generated_angle_rad), np.cos(generated_angle_rad), 0], 
                                            [0, 0, 1]]))
                p = p + median_point
                new_clip_points.append(p)

            new_clip_points = np.array(new_clip_points)
            new_clip_features = clip_features

        return new_clip_points, new_clip_features
                        
if __name__ == '__main__':
    dataset = Radar(root='/workspace/radar-nn-model/data/radar/train')
    clip_points, clip_features, label, video_idx, positive_points, positive_features = dataset[0]
    print(dataset.videos[1])
    print(len(dataset))
    print(dataset.labels)
