import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_npz_file, ann_npz_file, transform=None, target_transform=None):
        # Read data files
        self.images = np.load(img_npz_file)['arr_0']
        self.labels = np.load(ann_npz_file)['arr_0'].astype(np.long)
        # self.labels = np.eye(10)[self.labels]  # I found there was no need
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Read the specific image and its label
        image = self.images[index]
        label = self.labels[index]
        # Apply transform if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
