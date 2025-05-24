import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch
import random
import os


class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(image).type(torch.LongTensor)


# Dataset class for segmentation
class BreastCancerSegmentation(data.Dataset):
    def __init__(self, folder_path, masks, img_width, img_height):
        self.masks = [os.path.join(folder_path, mask) for mask in masks]
        self.img_width = img_width
        self.img_height = img_height

    def __getitem__(self, index):
        image = self.masks[index].replace("labels", "images")
        image = Image.open(image).convert("RGB").resize((self.img_width, self.img_height))
        label = Image.open(self.masks[index]).convert("L").resize((self.img_width, self.img_height))
        label = np.array(label, dtype=np.int64)
        image = np.array(image)

        label = (label > 127).astype(np.int64)

        tfms = transforms.Compose([
            transforms.ToTensor()
        ])

        y_transform = transforms.Compose([
            ToLabel(),
        ])
        img_new = tfms(image)
        label = y_transform(label)

        return img_new, label

    def __len__(self):
        return len(self.masks)

class BreastCancerImagesOnly(data.Dataset):
    def __init__(self, folder_path,  image_paths, img_width, img_height):
        """
        Dataset class for loading images without labels.
        
        Args:
            image_paths (list): List of paths to image files.
            img_width (int): Target width for resizing.
            img_height (int): Target height for resizing.
        """
        self.image_paths = [os.path.join(folder_path, path) for path in image_paths]
        self.img_width = img_width
        self.img_height = img_height

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB").resize((self.img_width, self.img_height))
        image = np.array(image)
        # Convert image to tensor
        tfms = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_new = tfms(image)
        return img_new, []

    def __len__(self):
        return len(self.image_paths)