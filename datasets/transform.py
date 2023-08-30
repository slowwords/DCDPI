from PIL import Image
import numpy as np
from torchvision import transforms


def image_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),   
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def image_crops(load_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=load_size, scale=(0.25, 0.8), ratio=(1, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
