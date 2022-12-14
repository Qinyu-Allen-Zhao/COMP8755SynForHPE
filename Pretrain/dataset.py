import os
import joblib
import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_path, ann_path, transform=None, target_transform=None):
        # Read data files
        self.images = []
        self.labels = []
        for sub_id in range(103):
            for img_id in range(500):
                image_file = os.path.join(img_path, "sub{}_{:0>8}.png".format(sub_id, img_id))
                self.images.append(image_file)
            smpls = joblib.load(os.path.join(ann_path, 'sub%d.pkl' % sub_id))['smpls']
            self.labels.append(smpls)
        self.labels = np.concatenate(self.labels, axis=0)
        print("Read images: ", len(self.images))
        print("Read poses: ", self.labels.shape)

        # self.labels = np.eye(10)[self.labels]  # I found there was no need
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Read the specific image and its label
        image_path = self.images[index]
        image = cv2.imread(image_path)
        label = self.labels[index].astype(np.float32)
        # Apply transform if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
