import os
from pathlib import Path
from cell_similarity.data.datasets import ImageDataset

def test():

    path_dataset_test = Path(os.getcwd()) / 'dataset_test'
    dataset = ImageDataset(path_dataset_test)

    assert dataset.__len__() == len([i for i in os.listdir(path_dataset_test) if i.endswith(('.png', '.jpg', '.jpeg', '.tiff'))])


if __name__ == '__main__':
    test()
