from .datasets import ImageDataset

def make_dataset(
        dataset_path:str,
        transform=None,
):
    dataset = ImageDataset(root=dataset_path, transform=transform)

    return dataset
