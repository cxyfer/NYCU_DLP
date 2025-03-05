import os
import torch
import shutil
import numpy as np

from PIL import Image
import cv2
import imutils
from tqdm import tqdm
from urllib.request import urlretrieve

from utils import *

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        if not os.path.exists(self.images_directory) or not os.path.exists(self.masks_directory):
            self.download(root)

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.mode == "train" and self.transform is not None: # transform only for training
            # sample = self.transform(**sample)
            sample = transform(sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask): # from TA
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self): # from TA
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root): # from TA

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        # sample["image"] = np.moveaxis(image, -1, 0).astype(np.float32)
        sample["image"] = (np.moveaxis(image, -1, 0) / 255.0).astype(np.float32)
        sample["mask"] = np.expand_dims(mask, 0).astype(np.float32)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def rotate(image, angle, center=None, scale=1.0):
    """
    Rotate the image by the given angle and center.
    If the center is None, the image center is used.

    Arguments:
        image: the image to be rotated
        angle: the angle to rotate the image
        center: the center of the rotation
        scale: the scale of the rotation
    Returns:
        rotated image
    """
    (h, w) = image.shape[:2] # grab the dimensions of the image

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def transform(sample):
    """
    Transform the image and mask.
    """
    deg = np.random.randint(-15, 16) # [-15, 15]
    flip1 = np.random.rand() # [0, 1]
    flip2 = np.random.rand() # [0, 1]

    for key in sample.keys():
        sample[key] = rotate(sample[key], deg)
        if flip1 > 0.5:  # flipping around the x-axis (vertically)
            sample[key] = cv2.flip(sample[key], 0)
        if flip2 > 0.5: # flipping around the y-axis (horizontally)
            sample[key] = cv2.flip(sample[key], 1)
    return sample

def load_dataset(data_path, mode):
    # implement the load dataset function here
    root = os.path.join(data_path, "oxford-iiit-pet") # 直到繳交時才發現還有一個 subfolder ...
    return SimpleOxfordPetDataset(root=root, mode=mode, transform=transform)

if __name__ == "__main__":
    dataset = load_dataset(data_path="./dataset", mode="train")
    idx = 16
    print(len(dataset))
    print(dataset[idx].keys())
    for mode in ["train", "valid", "test"]:
        dataset = load_dataset(data_path="./dataset", mode=mode)
        print(f"mode: {mode}, len: {len(dataset)}")
        print("/".join(dataset[idx].keys()), ", ".join([str(v.shape) for v in dataset[idx].values()]), sep=": ")
        show_dataset_sample(dataset[idx], f"{mode}[{idx}]")