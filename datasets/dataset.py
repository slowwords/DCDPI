import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from utils.canny import image_to_edge
from datasets.transform import mask_transforms, image_transforms, image_crops
from datasets.folder import make_dataset


class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root, load_size, sigma=2., mode='test'):
        super(ImageDataset, self).__init__()

        self.image_files = make_dataset(dir=image_root)
        self.mask_files = make_dataset(dir=mask_root)

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)

        self.sigma = sigma
        self.mode = mode

        self.load_size = load_size
        self.image_files_crops = image_crops(load_size)
        self.image_files_transforms = image_transforms(load_size)   
        self.mask_files_transforms = mask_transforms(load_size)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        if self.mode == 'train':
            is_random = random.randint(0, 3)
        else:
            is_random = 1
        if is_random == 0:
            image = self.image_files_crops(image.convert('RGB'))
        else:
            image = self.image_files_transforms(image.convert('RGB'))

        if self.mode == 'train':
            mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])
        else:
            mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        edge, gray_image = image_to_edge(image, sigma=self.sigma)

        return image, mask, edge, gray_image

    def __len__(self):

        return self.number_image

    def random_crop(self, img):
        width = 256
        height = 256
        max_x = img.shape[1] - width
        max_y = img.shape[0] - height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = img[y: y + height, x: x + width]

        return crop, height, width


def create_image_dataset(opts):

    image_dataset = ImageDataset(
        opts.image_root,
        opts.mask_root,
        opts.load_size,
        opts.sigma,
        opts.mode
    )

    return image_dataset
