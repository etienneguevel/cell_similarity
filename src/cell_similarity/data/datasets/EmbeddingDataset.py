import os

import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):

    def __init__(self, root: str):
        self.data = torch.load(os.path.join(root, 'embeddings.pt'))
        self.labels = torch.load(os.path.join(root, 'labels.pt'))

    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index: int):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label
