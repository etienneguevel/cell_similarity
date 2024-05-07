import os
from pathlib import Path
from cell_similarity.data.datasets import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def test():

    path_dataset_test = Path(os.getcwd()) / 'dataset_test'
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(path_dataset_test, transform=transform)

    assert dataset.__len__() == len([i for i in os.listdir(path_dataset_test) if i.endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    dataloader = DataLoader(dataset, batch_size=32)
    for i in dataloader:
        assert len(i) == 32
        break

if __name__ == '__main__':
    test()
