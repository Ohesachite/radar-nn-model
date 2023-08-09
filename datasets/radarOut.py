import numpy as np
from torch.utils.data import Dataset

class RadarOutput(Dataset):
    def __init__ (self, vid_labels, *probs):
        self.probs = {k: np.concatenate([prob[k] / np.sum(prob[k], axis=1, keepdims=True) for prob in probs], axis=1) for k in vid_labels.keys()}
        self.labels = vid_labels
        self.index_map = []
        for k, v in self.probs.items():
            for s in range(v.shape[0]):
                self.index_map.append((k, s))

    def __len__ (self):
        return len(self.index_map)

    def __getitem__ (self, idx):
        key, seg = self.index_map[idx]

        label = self.labels[key]
        output = self.probs[key][seg,:]

        return output.astype(np.float32), label

