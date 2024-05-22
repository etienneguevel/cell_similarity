from typing import List, Any, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from.decoders import ImageDataDecoder

class CustomDataset(Dataset):

    def __init__(self, images_list: List[str], targets_list: List[str], transform: Any = None, target_map: Dict[str, int]=None):
        self.images_list = images_list
        self.targets_list = targets_list
        self.transforms = transform
        self.target_map = target_map
        self.num_classes = len(target_map)

    def get_target(self, index: int):
        class_ = self.targets_list[index]
        class_id = torch.tensor(self.target_map[class_])

        return class_id


    def get_image_data(self, index: int):
        path = self.images_list[index]
        with open(path, 'rb') as f:
            image_data = f.read()

        return image_data
    

    def __getitem__(self, index: int) -> Any:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can nor read image for sample {index}") from e
        target = self.get_target(index)
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):

        return len(self.targets_list)
