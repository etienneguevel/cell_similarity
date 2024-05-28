import os

import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):

    def __init__(self, root: str, device:str="cpu"):
        self.data = torch.load(os.path.join(root, 'embeddings.pt'), map_location=torch.device(device))
        self.labels = torch.load(os.path.join(root, 'labels.pt'), map_location=torch.device(device))
        self.data_norm = self._normalize_data()

    def _normalize_data(self):
        m, std = torch.mean(self.data, dim=0), torch.std(self.data, dim=0)
        data_norm = (self.data - m) / std
        return data_norm

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index: int):
        sample = self.data_norm[index]
        label = self.labels[index]

        return sample, label
