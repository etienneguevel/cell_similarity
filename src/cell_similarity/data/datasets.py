import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_list = self._get_image_list()

    def _get_image_list(self):
        images = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    try:
                        Image.open(os.path.join(root, file))
                        images.append(os.path.join(root, file))
                    
                    except OSError:
                        print(f"Image at path {os.path.join(root, file)} could not be opened.")

        return images
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
    