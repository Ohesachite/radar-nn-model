import os
import sys
import numpy as np
import scipy as sp
from scipy.special import softmax
import random
from torch.utils.data import Dataset
import csv
import json

class Radar(Dataset):
    def __init__(self, root, mode=0, frames_per_clip=32, frame_interval=1, num_points=2048, mask_split=[1.0, 0.0, 0.0], decay=0.002, retclipnum=False):
        super(Radar, self).__init__()

        self.decay = decay
        self.retclipnum = retclipnum

        self.videos = []
        self.num_samples = []
        self.num_clips = []
        self.generate_positive = {}
        self.labels = []
        self.cycle_data = []
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

        segment_file = False
        if os.path.exists(os.path.join(root, "segment_boundaries.json")):
            boundary_file = open(os.path.join(root, "segment_boundaries.json"))
            boundaries = json.load(boundary_file)
            boundary_file.close()
            segment_file = True

        vid_index_num_map = {}
        vid_num_index_map = {}

        idx = 0
        for file_name in os.listdir(root):
            if file_name.endswith(".npy"):
                name_parts = file_name.split('.')[0].split('_')
                if name_parts[0] == "label":
                    point_clouds[(name_parts[1], name_parts[2])] = np.load(os.path.join(root, file_name))
                    self.videos.append(point_clouds[(name_parts[1], name_parts[2])])
                    self.num_samples.append(0)
                    self.labels.append(label_key[name_parts[1]])

                    vid_index_num_map[(index, label_key[name_parts[1]])] = name_parts[2]
                    vid_num_index_map[(name_parts[2], label_key[name_parts[1]])] = index

                    nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                    if mode == 0:
                        for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                            self.index_map.append((index, t))
                            if t < (nframes-frame_interval*(frames_per_clip-1)) * mask_split[0]:
                                self.train_indices.append(idx)
                            elif t < (nframes-frame_interval*(frames_per_clip-1)) * (mask_split[0] + mask_split[1]):
                                self.val_indices.append(idx)
                            else:
                                self.test_indices.append(idx)
                            self.num_samples[index] += 1
                            idx += 1

                    elif mode == 3 or mode == 5 or mode == 6:
                        if segment_file:
                            # boundaries[(name_parts[1], name_parts[2])] should be list of tuples (start, end, list with indices sorted in order of largest self correlation)
                            seg_num = 0
                            if retclipnum:
                                self.num_clips.append(0)
                                clip_num = 0
                            ssm = boundaries[name_parts[1]][name_parts[2]]
                            final_bounds = self.ssm_to_bounds(ssm, name_parts[1])

                            for start, end, order in final_bounds:
                                if end - start + 1 < frames_per_clip:
                                    if mode == 3 or mode == 5:
                                        chosen_frames = []
                                        nframes = frames_per_clip
                                        while end - start + 1 < nframes:
                                            chosen_frames += order
                                            nframes -= end - start + 1
                                        chosen_frames += order[:nframes]
                                        chosen_frames.sort()
                                    
                                        self.index_map.append((index, chosen_frames, seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                    elif mode == 6:
                                        randstart = np.random.randint(max(0, end-(frames_per_clip-1)), min(start+1, nframes-(frames_per_clip-1)))
                                        if retclipnum:
                                            self.index_map.append((index, list(range(randstart, randstart+frames_per_clip)), seg_num, clip_num))
                                            self.num_clips[index] += 1
                                            clip_num += 1
                                        else:
                                            self.index_map.append((index, list(range(randstart, randstart+frames_per_clip)), seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                else:
                                    if mode == 3:
                                        num_sects = (end - start + 1) // frames_per_clip + 1
                                        for i in range(num_sects):
                                            sect_start = start + (end - start + 1) * i // num_sects
                                            sect_end = start + (end - start + 1) * (i+1) // num_sects
                                            sect_order = [x for x in order if sect_start <= x < sect_end]
                                            chosen_frames = []
                                            nframes = frames_per_clip
                                            while sect_end - sect_start < nframes:
                                                chosen_frames += sect_order
                                                nframes -= sect_end - sect_start
                                            chosen_frames += sect_order[:nframes]
                                            chosen_frames.sort()

                                            self.index_map.append((index, chosen_frames, seg_num))
                                            self.train_indices.append(idx)
                                            self.test_indices.append(idx)
                                            idx += 1
                                    elif mode == 5:
                                        order.remove(start)
                                        order.remove(end)
                                        chosen_frames = order[:frames_per_clip-2]
                                        chosen_frames.sort()
                                        chosen_frames.insert(0, start)
                                        chosen_frames.append(end)

                                        self.index_map.append((index, chosen_frames, seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                    elif mode == 6:
                                        for t in range(start, end + 1 - frame_interval*(frames_per_clip-1)):
                                            if retclipnum:
                                                self.index_map.append((index, list(range(t, t+frames_per_clip)), seg_num, clip_num))
                                                self.num_clips[index] += 1
                                                clip_num += 1
                                            else:
                                                self.index_map.append((index, list(range(t, t+frames_per_clip)), seg_num))
                                            self.train_indices.append(idx)
                                            self.test_indices.append(idx)
                                            idx += 1

                                    else:
                                        raise ValueError("We shouldn't have arrived here")

                                self.num_samples[index] += 1
                                seg_num += 1

                            # for start, end, order in boundaries[name_parts[1]][name_parts[2]]:
                            #     if end - start + 1 < frames_per_clip:
                            #         if mode == 3 or mode == 5:
                            #             chosen_frames = []
                            #             nframes = frames_per_clip
                            #             while end - start + 1 < nframes:
                            #                 chosen_frames += order
                            #                 nframes -= end - start + 1
                            #             chosen_frames += order[:nframes]
                            #             chosen_frames.sort()
                                    
                            #             self.index_map.append((index, chosen_frames, seg_num))
                            #             self.train_indices.append(idx)
                            #             self.test_indices.append(idx)
                            #             idx += 1
                            #         elif mode == 6:
                            #             randstart = np.random.randint(max(0, end-(frames_per_clip-1)), min(start+1, nframes-(frames_per_clip-1)))
                            #             self.index_map.append((index, list(range(randstart, randstart+frames_per_clip)), seg_num))
                            #             self.train_indices.append(idx)
                            #             self.test_indices.append(idx)
                            #             idx += 1
                            #     else:
                            #         if mode == 3:
                            #             num_sects = (end - start + 1) // frames_per_clip + 1
                            #             for i in range(num_sects):
                            #                 sect_start = start + (end - start + 1) * i // num_sects
                            #                 sect_end = start + (end - start + 1) * (i+1) // num_sects
                            #                 sect_order = [x for x in order if sect_start <= x < sect_end]
                            #                 chosen_frames = []
                            #                 nframes = frames_per_clip
                            #                 while sect_end - sect_start < nframes:
                            #                     chosen_frames += sect_order
                            #                     nframes -= sect_end - sect_start
                            #                 chosen_frames += sect_order[:nframes]
                            #                 chosen_frames.sort()

                            #                 self.index_map.append((index, chosen_frames, seg_num))
                            #                 self.train_indices.append(idx)
                            #                 self.test_indices.append(idx)
                            #                 idx += 1
                            #         elif mode == 5:
                            #             order.remove(start)
                            #             order.remove(end)
                            #             chosen_frames = order[:frames_per_clip-2]
                            #             chosen_frames.sort()
                            #             chosen_frames.insert(0, start)
                            #             chosen_frames.append(end)

                            #             self.index_map.append((index, chosen_frames, seg_num))
                            #             self.train_indices.append(idx)
                            #             self.test_indices.append(idx)
                            #             idx += 1
                            #         elif mode == 6:
                            #             for t in range((end + 1 - start) - frame_interval*(frames_per_clip-1)):
                            #                 self.index_map.append((index, list(range(t, t+frames_per_clip)), seg_num))
                            #                 self.train_indices.append(idx)
                            #                 self.test_indices.append(idx)
                            #                 idx += 1

                            #         else:
                            #             raise ValueError("We shouldn't have arrived here")

                            #     self.num_samples[index] += 1
                            #     seg_num += 1

                        else:
                            raise ValueError("A file named segment_boundaries.json is required for mode 3 to work")
                    elif mode == 4:
                        nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                        for t in range(0, nframes-frame_interval*(frames_per_clip-1), frame_interval*frames_per_clip):
                            self.index_map.append((index, t))
                            if t < (nframes-frame_interval*(frames_per_clip-1)) * mask_split[0]:
                                self.train_indices.append(idx)
                            elif t < (nframes-frame_interval*(frames_per_clip-1)) * (mask_split[0] + mask_split[1]):
                                self.val_indices.append(idx)
                            else:
                                self.test_indices.append(idx)
                            self.num_samples[index] += 1
                            idx += 1

                    else:
                        raise ValueError("You need cycle data for mode 1 or 2. This would normally be found in npz files rather than npy")

                    index += 1

            elif file_name.endswith(".npz"):
                name_parts = file_name.split('.')[0].split('_')
                if name_parts[0] == "label":
                    loaded_data = np.load(os.path.join(root, file_name), allow_pickle=True).item()
                    point_clouds[(name_parts[1], name_parts[2])] = loaded_data['Point cloud']
                    self.videos.append(point_clouds[(name_parts[1], name_parts[2])])
                    self.num_samples.append(0)
                    self.labels.append(label_key[name_parts[1]])

                    vid_index_num_map[(index, label_key[name_parts[1]])] = name_parts[2]
                    vid_num_index_map[(name_parts[2], label_key[name_parts[1]])] = index

                    if 'Cycle data' in loaded_data.keys():
                        self.cycle_data.append(loaded_data['Cycle data'])
                    elif mode == 1 or mode == 2:
                        assert(False, "Attempting to load data with mode 1 or 2 but missing some cycle data")

                    nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                    if mode == 0:
                        for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                            self.index_map.append((index, t))
                            if t < (nframes-frame_interval*(frames_per_clip-1)) * mask_split[0]:
                                self.train_indices.append(idx)
                            elif t < (nframes-frame_interval*(frames_per_clip-1)) * (mask_split[0] + mask_split[1]):
                                self.val_indices.append(idx)
                            else:
                                self.test_indices.append(idx)
                            self.num_samples[index] += 1
                            idx += 1
                            
                    elif mode == 2: # No index mapping for mode 1 supported, mode 2 produces synthetically generated data
                        # f - frame skips, c - count padding, r - number of repeats
                        skips = [(f, c, r) for f in [1,2] for c in [0,-1,1] for r in [1,2,3]]
                        start_buffer, end_buffer, init_count = self.cycle_data[index]
                        if init_count == 0:
                            self.index_map.append((index, 0, range(nframes)))
                        else:
                            init_start_frame = start_buffer
                            init_end_frame = nframes - end_buffer
                            init_avg_period = (init_end_frame - init_start_frame) / init_count
                            indices = list(range(nframes))
                            for f, c, r in skips:
                                # each entry in index map should be a tuple containing vid_index, count, and frame order
                                # count padding first, then frame_skipping, then repeats
                                rep_indices = indices[init_start_frame:init_end_frame]
                                total_period = int(init_avg_period * (init_count + c))
                                if c == -1:
                                    rep_indices = rep_indices[:total_period]
                                elif c == 1:
                                    rep_indices = rep_indices + rep_indices[:(total_period - (init_end_frame - init_start_frame))]

                                repeat_indices = []
                                for start in range(f):
                                    repeat_indices = repeat_indices + rep_indices[start::f]
                                repeat_indices = repeat_indices * r

                                final_indices = indices[:init_start_frame] + repeat_indices + indices[init_end_frame:]
                                self.index_map.append((index, (init_count + c) * f * r, final_indices))

                    elif mode == 3 or mode == 5 or mode == 6:
                        if segment_file:
                            # boundaries[(name_parts[1], name_parts[2])] should be list of tuples (start, end, list with indices sorted in order of largest self correlation)
                            seg_num = 0
                            ssm = boundaries[name_parts[1]][name_parts[2]]
                            final_bounds = self.ssm_to_bounds(ssm, name_parts[1])

                            for start, end, order in final_bounds:
                                if end - start + 1 < frames_per_clip:
                                    if mode == 3 or mode == 5:
                                        chosen_frames = []
                                        nframes = frames_per_clip
                                        while end - start + 1 < nframes:
                                            chosen_frames += order
                                            nframes -= end - start + 1
                                        chosen_frames += order[:nframes]
                                        chosen_frames.sort()
                                    
                                        self.index_map.append((index, chosen_frames, seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                    elif mode == 6:
                                        randstart = np.random.randint(max(0, end-(frames_per_clip-1)), min(start+1, nframes-(frames_per_clip-1)))
                                        self.index_map.append((index, list(range(randstart, randstart+frames_per_clip)), seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                else:
                                    if mode == 3:
                                        num_sects = (end - start + 1) // frames_per_clip + 1
                                        for i in range(num_sects):
                                            sect_start = start + (end - start + 1) * i // num_sects
                                            sect_end = start + (end - start + 1) * (i+1) // num_sects
                                            sect_order = [x for x in order if sect_start <= x < sect_end]
                                            chosen_frames = []
                                            nframes = frames_per_clip
                                            while sect_end - sect_start < nframes:
                                                chosen_frames += sect_order
                                                nframes -= sect_end - sect_start
                                            chosen_frames += sect_order[:nframes]
                                            chosen_frames.sort()

                                            self.index_map.append((index, chosen_frames, seg_num))
                                            self.train_indices.append(idx)
                                            self.test_indices.append(idx)
                                            idx += 1
                                    elif mode == 5:
                                        order.remove(start)
                                        order.remove(end)
                                        chosen_frames = order[:frames_per_clip-2]
                                        chosen_frames.sort()
                                        chosen_frames.insert(0, start)
                                        chosen_frames.append(end)

                                        self.index_map.append((index, chosen_frames, seg_num))
                                        self.train_indices.append(idx)
                                        self.test_indices.append(idx)
                                        idx += 1
                                    elif mode == 6:
                                        for t in range(start, end + 1 - frame_interval*(frames_per_clip-1)):
                                            self.index_map.append((index, list(range(t, t+frames_per_clip)), seg_num))
                                            self.train_indices.append(idx)
                                            self.test_indices.append(idx)
                                            idx += 1

                                    else:
                                        raise ValueError("We shouldn't have arrived here")

                                self.num_samples[index] += 1
                                seg_num += 1

                        else:
                            raise ValueError("A file named segment_boundaries.json is required for mode 3 to work")

                    elif mode == 4:
                        nframes = int(np.amax(point_clouds[(name_parts[1], name_parts[2])][:,0])) + 1
                        for t in range(0, nframes-frame_interval*(frames_per_clip-1), frame_interval*frames_per_clip):
                            self.index_map.append((index, t))
                            if t < (nframes-frame_interval*(frames_per_clip-1)) * mask_split[0]:
                                self.train_indices.append(idx)
                            elif t < (nframes-frame_interval*(frames_per_clip-1)) * (mask_split[0] + mask_split[1]):
                                self.val_indices.append(idx)
                            else:
                                self.test_indices.append(idx)
                            self.num_samples[index] += 1
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
        self.mode = mode

    def __len__(self):
        if self.mode == 1:
            return len(self.videos)
        else:
            return len(self.index_map)
        
    def __getitem__(self, idx):
        if self.mode == 0 or self.mode == 4:
            index, t = self.index_map[idx]

            video = self.videos[index]
            ns = self.num_samples[index]
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

            if idx in self.train_indices:
                # scale height and translate body
                scale = np.random.uniform(0.9, 1.1, size=3)
                offset = np.random.uniform(-0.5, 0.5, size=2)
                clip_points = clip_points * scale
                clip_points[:,:,:2] = clip_points[:,:,:2] + offset

            positive_clip_points, positive_clip_features = self.generate_positive_sample(clip_points, clip_features, idx)

            return clip_points.astype(np.float32), clip_features.astype(np.float32), label, index, t, ns, positive_clip_points.astype(np.float32), positive_clip_features.astype(np.float32)
        
        elif self.mode == 1:
            video = self.videos[idx]

            sampled_video = []
            n_frames = int(np.amax(video[:,0]) + 1)
            for i in range(n_frames):
                frame = video[video[:,0] == i, 2:]
                sampled_video.append(frame)

            for i, p in enumerate(sampled_video):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                sampled_video[i] = p[r,:]

            sampled_video = np.array(sampled_video)

            vid_points = sampled_video[:,:,:3]
            vid_features = sampled_video[:,:,3:]
            vid_features = np.swapaxes(vid_features, 1, 2)

            scale = np.random.uniform(0.9, 1.1, size=3)
            offset = np.random.uniform(-0.5, 0.5, size=2)
            clip_points = clip_points * scale
            clip_points[:,:,:2] = clip_points[:,:,:2] + offset

            start_buffer, end_buffer, count = self.cycle_data[idx]
            start_frame = start_buffer
            end_frame = n_frames - end_buffer

            y1, y2, avg_period, optimal_stride = self.get_repetition_counting_target(n_frames, self.frames_per_clip, start_frame, end_frame, count)

            return vid_points.astype(np.float32), vid_features.astype(np.float32), (y1.astype(np.float32), y2.astype(np.float32), avg_period, optimal_stride, count)

        elif self.mode == 2:
            index, count, frame_indices = self.index_map[idx]
            video = self.videos[index]
            
            sampled_video = []
            n_frames = len(frame_indices)
            for i in frame_indices:
                frame = video[video[:,0] == i, 2:]
                sampled_video.append(frame)

            for i, p in enumerate(sampled_video):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                sampled_video[i] = p[r,:]

            sampled_video = np.array(sampled_video)

            vid_points = sampled_video[:,:,:3]
            vid_features = sampled_video[:,:,3:]
            vid_features = np.swapaxes(vid_features, 1, 2)

            if idx in self.train_indices:
                # scale height and translate body
                scale = np.random.uniform(0.9, 1.1, size=3)
                offset = np.random.uniform(-0.5, 0.5, size=2)
                clip_points = clip_points * scale
                clip_points[:,:,:2] = clip_points[:,:,:2] + offset

            start_buffer, end_buffer, _ = self.cycle_data[index]
            start_frame = start_buffer
            end_frame = n_frames - end_buffer

            y1, y2, avg_period, optimal_stride = self.get_repetition_counting_target(n_frames, self.frames_per_clip, start_frame, end_frame, count)

            return vid_points.astype(np.float32), vid_features.astype(np.float32), (y1.astype(np.float32), y2.astype(np.float32), avg_period, optimal_stride, count)
        
        elif self.mode == 3 or self.mode == 5 or self.mode == 6:
            if self.retclipnum:
                index, indices, seg_num, clip_num = self.index_map[idx]
                nc = self.num_clips[index]
            else:
                index, indices, seg_num = self.index_map[idx]
            video = self.videos[index]
            ns = self.num_samples[index]
            label = self.labels[index]
    
            clip = []
            for i in indices:
                frame = video[video[:,0] == i, 2:]
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

            if idx in self.train_indices:
                # scale height and translate body
                scale = np.random.uniform(0.9, 1.1, size=3)
                offset = np.random.uniform(-0.5, 0.5, size=2)
                clip_points = clip_points * scale
                clip_points[:,:,:2] = clip_points[:,:,:2] + offset

            positive_clip_points, positive_clip_features = self.generate_positive_sample(clip_points, clip_features, idx)

            if self.retclipnum:
                return clip_points.astype(np.float32), clip_features.astype(np.float32), label, index, seg_num, ns, positive_clip_points.astype(np.float32), positive_clip_features.astype(np.float32), clip_num, nc
            else:
                return clip_points.astype(np.float32), clip_features.astype(np.float32), label, index, seg_num, ns, positive_clip_points.astype(np.float32), positive_clip_features.astype(np.float32)
        
        else:
            assert(False, "Invalid mode")

    def generate_positive_sample(self, clip_points, clip_features, idx):
        if idx not in self.train_indices:
            return clip_points, clip_features

        index = self.index_map[idx][0]

        if self.generate_positive[index][0] != -1:
            index = random.choice(self.generate_positive[index])
            nframes = int(np.amax(self.videos[index][:,0])) + 1
            t = random.choice(range(0, nframes-self.frame_interval*(self.frames_per_clip-1)))

            video = self.videos[index]
    
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

            # scale and translate body
            scale = np.random.uniform(0.9, 1.1, size=3)
            offset = np.random.uniform(-0.5, 0.5, size=2)
            new_clip_points = new_clip_points * scale
            new_clip_points[:,:,:2] = new_clip_points[:,:,:2] + offset

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

            # scale and translate body
            scale = np.random.uniform(0.9, 1.1, size=3)
            offset = np.random.uniform(-0.5, 0.5, size=2)
            new_clip_points = new_clip_points * scale
            new_clip_points[:,:,:2] = new_clip_points[:,:,:2] + offset

        return new_clip_points, new_clip_features

    def get_repetition_counting_target(self, n_frames, sample_size, start_frame, end_frame, count, distance_weight=2):
        assert(0 <= start_frame < end_frame <= n_frames)

        if count != 0:
            avg_period = (end_frame - start_frame) / count
            optimal_stride = 1
            if avg_period > (sample_size//2):
                optimal_stride = int(np.ceil(avg_period / (sample_size//2)))
                avg_period = avg_period / optimal_stride

            y1 = np.array([[- distance_weight * (n - avg_period)**2 for n in range(1, sample_size//2+1)]])
            y1 = np.repeat(y1, n_frames, axis=0)
            y2 = np.array([1 if start_frame <= n <= end_frame else 0 for n in range(n_frames)])
            
            for n in range(n_frames):
                if y2[n] == 1:
                    y1[n,:] = softmax(y1[n,:])
                else:
                    y1[n,:] = np.zeros(sample_size//2)
                    y1[n,0] = 1

            return y1, y2, avg_period, optimal_stride

        else:
            y1 = np.transpose(np.append(np.ones((1,n_frames)), np.zeros(((sample_size//2) - 1, n_frames)), axis=0))
            y2 = np.zeros(n_frames)

            return y1, y2, 1.0, 1

    def ssm_to_bounds(self, ssm, label):
        ssm = np.array(ssm)

        bounds = self.find_boundaries(ssm)
        bounds = self.dynamic_bound_combination(bounds, label)

        final_bounds = []

        for i in range(len(bounds)):
            bound_l, bound_r = bounds[i]

            region_in_bounds_order = np.argsort(np.diag(ssm)[bound_l:bound_r+1])[::-1]
            region_in_bounds_order += bound_l
            region_in_bounds_order = region_in_bounds_order.tolist()

            final_bounds += [(int(bound_l), int(bound_r), region_in_bounds_order)]

        return final_bounds

    def find_boundaries(self, ssm_peak_matrix):
        num_peaks = ssm_peak_matrix.shape[0]
        assert ssm_peak_matrix.shape[1] == num_peaks
        
        peaks_in_order = np.argsort(np.diag(ssm_peak_matrix))[::-1]
        
        candidate_regions = []
        
        for peak in peaks_in_order:
            
            if len(candidate_regions) == 0:
                candidate_regions.append([peak])
                continue
                
            if peak < candidate_regions[0][0]:
                if ssm_peak_matrix[peak, candidate_regions[0][0]] < 0.0:
                    candidate_regions.insert(0, [peak])
                else:
                    candidate_regions[0].insert(0, peak)
            elif peak > candidate_regions[-1][-1]:
                if ssm_peak_matrix[peak, candidate_regions[-1][-1]] < 0.0:
                    candidate_regions.append([peak])
                else:
                    candidate_regions[-1].append(peak)
            else:
                for ri in range(0, len(candidate_regions)):
                    if ri != 0 and candidate_regions[ri-1][-1] < peak < candidate_regions[ri][0]:
                        corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri-1][-1]]
                        corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][0]]
                        
                        if corr_peak_l < 0.0 and corr_peak_r < 0.0:
                            candidate_regions.insert(ri, [peak])
                        elif corr_peak_r > corr_peak_l:
                            candidate_regions[ri].insert(0, peak)
                        else:
                            candidate_regions[ri-1].append(peak)
                            
                        break
                    elif candidate_regions[ri][0] < peak < candidate_regions[ri][-1]:
                        for wri in range(1, len(candidate_regions[ri])):
                            if candidate_regions[ri][wri-1] < peak < candidate_regions[ri][wri]:
                                corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri][wri-1]]
                                corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][wri]]
                                
                                new_region_l = candidate_regions[ri][:wri]
                                new_region_r = candidate_regions[ri][wri:]
                                
                                if corr_peak_l < 0.0 and corr_peak_r < 0.0:
                                    del candidate_regions[ri]
                                    candidate_regions.insert(ri, new_region_r)
                                    candidate_regions.insert(ri, [peak])
                                    candidate_regions.insert(ri, new_region_l)
                                elif corr_peak_r > 0.0 and corr_peak_l < 0.0:
                                    del candidate_regions[ri][:wri]
                                    candidate_regions[ri].insert(0, peak)
                                    candidate_regions.insert(ri, new_region_l)
                                elif corr_peak_r < 0.0 and corr_peak_l > 0.0:
                                    del candidate_regions[ri][:wri]
                                    candidate_regions.insert(ri, new_region_l)
                                    candidate_regions[ri].append(peak)
                                else:
                                    candidate_regions[ri].insert(wri, peak)
                                        
                                break
                        break
        
        num_regions = len(candidate_regions)
        region_score = np.zeros((num_regions, num_regions))
        
        for i in range(num_regions):
            region_r = ssm_peak_matrix[candidate_regions[i],:]
            for j in range(num_regions):
                region_rc = region_r[:,candidate_regions[j]]
                region_score[i,j] = np.sum(region_rc)
        
        def best_boundaries(region_scores, decay_factor=self.decay):
            num_regions = len(region_scores)
            seg_scores = np.zeros((num_regions // 2 + 1, num_regions))
            best_prev_up_index = -np.ones((num_regions // 2 + 1, num_regions, 2), dtype=int)
            
            memoized_block_scores = np.zeros(region_scores.shape)
            
            for pot_up in range(num_regions):
                for pot_down in range(num_regions):
                    memoized_block_scores[pot_up, pot_down] = region_scores[pot_up, pot_up] + region_scores[pot_down, pot_down] - 2*region_scores[pot_up, pot_down]
            
            for pot_down in range(1, num_regions):
                best_up = np.argmax(memoized_block_scores[:pot_down, pot_down])
                best_prev_up_index[1, pot_down, 1] = best_up
                seg_scores[1, pot_down] = memoized_block_scores[best_up, pot_down]
                
            for segn in range(2, num_regions // 2 + 1):
                for pot_down in range(segn * 2 - 1, num_regions):
                    up_vect = np.repeat(np.expand_dims(memoized_block_scores[(segn-1)*2:pot_down, pot_down], axis=0), pot_down-(segn-1)*2, axis=0)
                    prev_vect = np.repeat(np.expand_dims(seg_scores[segn-1, (segn-1)*2-1:pot_down-1], axis=1), pot_down-(segn-1)*2, axis=1)
                    sum_matrix = np.triu(up_vect + prev_vect)
                    
                    unadj_prev_up = np.unravel_index(np.argmax(sum_matrix), sum_matrix.shape)
                    best_prev_up = (unadj_prev_up[0] + (segn-1)*2 - 1, unadj_prev_up[1] + (segn-1)*2)
                    best_prev_up_index[segn, pot_down, :] = best_prev_up
                    
                    seg_scores[segn, pot_down] = seg_scores[segn-1, best_prev_up[0]] + memoized_block_scores[best_prev_up[1], pot_down]
                    
            for segn in range(num_regions // 2 + 1):
                seg_scores[segn,:] *=  (1 - decay_factor)**segn
                
            best_seg_index = np.unravel_index(np.argmax(seg_scores), seg_scores.shape)
            best_seg = []
            
            curr_down = best_seg_index[1]
            for segn in range(best_seg_index[0], 0, -1):
                curr_prev_up = best_prev_up_index[segn, curr_down, :]
                best_seg.append([curr_prev_up[1], curr_down])
                curr_down = curr_prev_up[0]
            
            return best_seg
        
        rn_boundaries = best_boundaries(region_score)
        
        peak_boundaries = []
        
        if len(rn_boundaries) == 0:
            peak_boundaries += [[0, num_peaks-1]]
        else:
            for rnb in rn_boundaries[::-1]:
                peak_boundaries += [[candidate_regions[rnb[0]][0], candidate_regions[rnb[1]][-1]]]
            
        return peak_boundaries

    def dynamic_bound_combination(self, boundaries, label):
        for i in range(len(boundaries)-1, -1, -1):
            bound_l, bound_r = boundaries[i]
            if bound_r - bound_l < 8:
                del boundaries[i]
            
        bound_sizes = [bound_r - bound_l + 1 for bound_l, bound_r in boundaries]
        
        new_boundaries = []
        if label == 'sq':
            bound_size_expectation = 22
        elif label == 'aud':
            bound_size_expectation = 60
        elif label == 'lud':
            bound_size_expectation = 24
        elif label == 'pu':
            bound_size_expectation = int(sum(bound_sizes) * 3 / len(boundaries) / 4)
        elif label == 'su':
            bound_size_expectation = 23
        else:
            bound_size_expectation = 0
                                        
        combo_map = [0] * len(boundaries)
        lpointer = 0
        rpointer = len(boundaries) - 1
        while lpointer < len(boundaries) and bound_sizes[lpointer] >= bound_size_expectation:
            lpointer += 1
        while rpointer >= 0 and bound_sizes[rpointer] >= bound_size_expectation:
            rpointer -= 1
            
        while lpointer < rpointer:
            if bound_sizes[lpointer] > bound_sizes[rpointer]:
                if rpointer < len(boundaries) - 1 and combo_map[rpointer+1] == 0:
                    combo_map[rpointer] = 1
                    combo_map[rpointer+1] = -1
                    rpointer -= 1
                else:
                    combo_map[rpointer] = -1
                    combo_map[rpointer-1] = 1
                    rpointer -= 2
                
                while rpointer >= 0 and bound_sizes[rpointer] >= bound_size_expectation:
                    rpointer -= 1
                    
            else:
                if lpointer > 0 and combo_map[lpointer-1] == 0:
                    combo_map[lpointer] = -1
                    combo_map[lpointer-1] = 1
                    lpointer += 1
                else:
                    combo_map[lpointer] = 1
                    combo_map[lpointer+1] = -1
                    lpointer += 2
                
                while lpointer < len(boundaries) and bound_sizes[lpointer] >= bound_size_expectation:
                    lpointer += 1
            
        if lpointer == rpointer and bound_sizes[lpointer] < bound_size_expectation:
            if lpointer != 0 and combo_map[lpointer-1] == 0 and rpointer != len(boundaries) - 1 and combo_map[rpointer+1] == 0:
                if bound_sizes[lpointer-1] > bound_sizes[rpointer+1]:
                    combo_map[rpointer] = 1
                    combo_map[rpointer+1] = -1
                else:
                    combo_map[lpointer] = -1
                    combo_map[lpointer-1] = 1
            elif lpointer != 0 and combo_map[lpointer-1] == 0:
                combo_map[lpointer] = -1
                combo_map[lpointer-1] = 1
            elif rpointer != len(boundaries) - 1 and combo_map[rpointer+1] == 0:
                combo_map[rpointer] = 1
                combo_map[rpointer+1] = -1
            
        i = 0
        while i < len(boundaries):
            if combo_map[i] == 1:
                new_boundaries.append([boundaries[i][0], boundaries[i+1][1]])
                i += 2
            elif combo_map[i] == 0:
                new_boundaries.append([boundaries[i][0], boundaries[i][1]])
                i += 1

        return new_boundaries

                        
if __name__ == '__main__':
    dataset = Radar(root='/home/alan/Documents/radar-nn-model/data/radar/train_2seen', mode=0, frames_per_clip=24, num_points=1024, decay=0.02)
    # for vid, points, label, index, start, vid_len, _, _ in dataset:
    #     print(vid.shape, points.shape, label, index, start, vid_len)
    print(len(dataset))
    nums_of_videos = [0] * dataset.num_classes
    nums_samples = [0] * dataset.num_classes
    for n in range(len(dataset.labels)):
        label = dataset.labels[n]
        samples = dataset.num_samples[n]
        if dataset.generate_positive[n][0] == -1:
            nums_of_videos[label] += 1
            nums_samples[label] += samples
        nums_of_videos[label] += 1
        nums_samples[label] += samples
    nums_of_clips = [0] * dataset.num_classes
    # for index, _, segnum in dataset.index_map:
    #     nums_of_clips[dataset.labels[index]] += 1
    for index, _ in dataset.index_map:
        if dataset.generate_positive[index][0] == -1:
            nums_of_clips[dataset.labels[index]] += 1
        nums_of_clips[dataset.labels[index]] += 1
    
    print(nums_of_videos)
    print(nums_of_clips)
    print(nums_samples)

    print(dataset.labels)
    
