import os
import sys
import numpy as np
from torch.utils.data import Dataset
import csv
import json

class Radar(Dataset):
    def __init__(self, root, frames_per_clip=32, frame_interval=1, num_points=2048, train=True):
        super(Radar, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
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

        for file_name in os.listdir(root):
            if file_name.endswith(".npy"):
                name_parts = file_name.split('.')[0].split('_')
                if name_parts[0] == "label":
                    point_clouds[(name_parts[1], name_parts[2])] = np.load(os.path.join(root, file_name))
                    self.videos.append(point_clouds[(name_parts[1], name_parts[2])])
                    self.labels.append(label_key[name_parts[1]])

                    nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                    for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                        self.index_map.append((index, t))
                    index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.train = train
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

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=5)
            clip = clip * scales

        clip_points = clip[:,:,:3]
        clip_features = clip[:,:,3:]
        clip_features = np.swapaxes(clip_features, 1, 2)

        return clip_points.astype(np.float32), clip_features.astype(np.float32), label, index
                        
if __name__ == '__main__':
    dataset = Radar(root='../data/radar/test', train=True)
    for i in range(len(dataset)):
        clip_points, clip_features, label, video_idx = dataset[i]
        print(clip_points.shape, clip_features.shape, label, video_idx)
    print(dataset.labels)
