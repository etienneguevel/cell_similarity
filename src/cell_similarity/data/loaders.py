import os
from typing import List

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .datasets import CustomDataset
from .transforms import make_classification_train_transform, make_classification_eval_transform

def make_datasets(
        root: str, 
        val_split: float = 0.1, 
        test_split: float = 0.1,
):

    _ = [[os.path.basename(dirpath), os.path.join(dirpath, i)] for dirpath, _, im in os.walk(root) for i in im if i.endswith(('.jpg', '.png', '.jpeg'))]
    targets_list = [temp[0] for temp in _]
    images_list = [temp[1] for temp in _]
    target_dict = {k: i for i, k in enumerate(os.listdir(root))}
    
    _x, images_test, _y, targets_test = train_test_split(images_list, targets_list, test_size=test_split, stratify=targets_list)
    images_train, images_val, targets_train, targets_val = train_test_split(_x, _y, test_size=val_split / (1-test_split), stratify=_y)

    datasets = {
        "train": CustomDataset(images_list=images_train, targets_list=targets_train, transform=make_classification_train_transform(), target_map=target_dict),
        "validation": CustomDataset(images_list=images_val, targets_list=targets_val, transform=make_classification_eval_transform(), target_map=target_dict),
        "test": CustomDataset(images_list=images_test, targets_list=targets_test, transform=make_classification_eval_transform(), target_map=target_dict)
    }

    return datasets

def make_dataloaders(
        datasets: List[Dataset],
        batch_size: int,
):
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x=="train")) for x in ["train", "validation", "test"]
    }

    return dataloaders

