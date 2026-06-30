import torch
from torch.utils.data import Dataset
import numpy as np


class HSSLDataset(Dataset):
    """Memory-efficient HSSL dataset: generates windows on-the-fly, stores only index mapping.
    
    Instead of storing all windows (5.3 GB for 994k windows), stores only
    (data_idx, start_sample) tuples — ~16 MB.
    """
    def __init__(self, data_list, window_size=700, overlap=0.5):
        self.data_list = data_list
        self.window_size = window_size
        self.step = max(1, int(window_size * (1 - overlap)))
        
        # Build index mapping: (data_idx, start_sample)
        self.indices = []
        for d_idx, d in enumerate(data_list):
            n = d["features"].shape[0]
            for start in range(0, n - window_size + 1, self.step):
                self.indices.append((d_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, start = self.indices[idx]
        feat = self.data_list[d_idx]["features"]
        w = feat[start:start + self.window_size]
        # HSSL expects (Channels, Sequence_Length)
        w_t = torch.tensor(w, dtype=torch.float32).transpose(0, 1)
        return w_t


class WindowLabelDataset(Dataset):
    """Memory-efficient window dataset that returns (window, label).
    
    Used by embedding generators (HSSL, SSL) to extract windows on-the-fly.
    """
    def __init__(self, features, labels=None, window_size=700, overlap=0.5):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.step = max(1, int(window_size * (1 - overlap)))
        
        # Build index mapping
        self.indices = list(range(0, len(features) - window_size + 1, self.step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        w = self.features[start:start + self.window_size]
        w_t = torch.tensor(w, dtype=torch.float32)
        
        if self.labels is not None:
            w_lbls = self.labels[start:start + self.window_size]
            val, counts = np.unique(w_lbls, return_counts=True)
            lbl = val[np.argmax(counts)]
            return w_t, torch.tensor(lbl, dtype=torch.long)
        
        return w_t