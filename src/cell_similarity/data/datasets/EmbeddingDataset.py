import os

import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):

    def __init__(self, root: str, device:str="cpu"):
        self.data = torch.load(os.path.join(root, 'embeddings.pt'), map_location=torch.device(device))
        self.labels = torch.load(os.path.join(root, 'labels.pt'), map_location=torch.device(device))

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index: int):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label
