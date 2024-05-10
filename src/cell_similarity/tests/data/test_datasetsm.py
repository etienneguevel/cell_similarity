import os
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cell_similarity.data.datasets import ImageDataset

base_path = Path(os.getcwd())
dirs = ['dataset_test', 'dataset_bis']


def test():

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.226])
    ])

    dataset = ImageDataset(root=dirs, transform=transform)
    
    expected_length = len([f for d in dirs for d, _, files in os.walk(d) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    assert dataset.__len__() == expected_length

    dataloader = DataLoader(dataset, batch_size=32)
    for i in dataloader:
        assert len(i)==32
        break

if __name__ == '__main__':
    test()
    
