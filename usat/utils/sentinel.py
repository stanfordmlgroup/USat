import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Callable

class ConsistentRandomCrop(T.RandomCrop):
    def __init__(self, sizes, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        # NOTE: Assume that sizes are sorted from smallest to largest img. This
        # is important to ensure that the crop values are truly equivalent across images
        self.smallest_size = sizes[0]
        self.sizes = [tuple(T.transforms._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")) for size in sizes]
        self.count = 0
        super().__init__(self.smallest_size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.sizes[self.count][1]:
            padding = [self.sizes[self.count][1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.sizes[self.count][0]:
            padding = [0, self.sizes[self.count][0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        if self.count == 0:
            self.ijhw = self.get_params(img, self.sizes[self.count])
            i, j, h, w = self.ijhw
        else:
            ratio_h = int(self.sizes[self.count][0] / self.smallest_size)
            ratio_w = int(self.sizes[self.count][1] / self.smallest_size)
            i, j = self.ijhw[0] * ratio_h, self.ijhw[1] * ratio_w
            h, w = self.sizes[self.count]

        self.count += 1
        if self.count % len(self.sizes) == 0:
            self.count = 0

        return F.crop(img, i, j, h, w)

class ConsistentRadomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, num_images, p=0.5):
        super().__init__(p)
        self.num_images = num_images
        self.count = 0

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.count == 0:
            self.rand_val = torch.rand(1)

        self.count += 1
        if self.count % self.num_images == 0:
            self.count = 0
    
        if self.rand_val < self.p:
            return F.hflip(img)
        return img


class ConsistentRadomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, num_images, p=0.5):
        super().__init__(p)
        self.num_images = num_images
        self.count = 0

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.count == 0:
            self.rand_val = torch.rand(1)

        self.count += 1
        if self.count % self.num_images == 0:
            self.count = 0
    
        if self.rand_val < self.p:
            return F.vflip(img)
        return img


class ConsistentRadomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, num_images, p=0.5):
        super().__init__(p)
        self.num_images = num_images
        self.count = 0

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.count == 0:
            self.rand_val = torch.rand(1)

        self.count += 1
        if self.count % self.num_images == 0:
            self.count = 0
    
        if self.rand_val < self.p:
            return F.vflip(img)
        return img


class SentinelNormalize(object):
    """ Normalization for Sentinel-2 data adapted from:
    https://github.com/sustainlab-group/SatMAE/blob/117135b3354fa70a81df453be7f34e3da8b36032/util/datasets.py#L349-L363
    """
    def __init__(self, mean, std, scale = 255.0):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.scale = scale

    def __call__(self, img, *args, **kwargs):
        min_val = self.mean - 2 * self.std
        max_val = self.mean + 2 * self.std

        img = (img - min_val) / (max_val - min_val)

        if self.scale == 1:
            img = np.clip(img, 0, 1)
        else:
            img *= self.scale
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img


class MinMaxNormalize(object):
    def __init__(self, min_val, max_val, scale=255):
        self.min = np.array(min_val)
        self.max = np.array(max_val)
        self.interval = self.max - self.min
        self.scale = scale

    def __call__(self, img, *args, **kwargs):
        img = (img - self.min) / self.interval * self.scale
        img = np.clip(img, 0, self.scale).astype(np.uint8)
        return img

 
def get_satmae_transforms(
        split: str, 
        image_size: int,
    ) -> List[Callable]:
    """"
    Get a set of transforms to apply to a sentinel dataset (common among
    BigEarthNet and EuroSAT)
    
    satmae_transform: Whether to match how SatMAE does its transforms
    standardize: Whether to apply transforms.normalize
    split: 'train' or 'val'
    mean: List of means for all raw channels
    std: List of std devs for all raw channels
    scaled_mean: List of means computed on SatMAE normalized values
    scaled_std: List of Std Devs computed on SatMAE normalized values
    image_size: Height (and width) to scale image to
    to_tensor: Whether to include a ToTensor Transform
    """
    img_transforms = []

    if split == 'train':
        img_transforms.append(
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC))
        img_transforms.append(T.RandomHorizontalFlip())
    else:
        crop_pct = 224 / 256 if image_size <= 224 else 1.0
        size = int(image_size / crop_pct)
        img_transforms.append(
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC))
        img_transforms.append(T.CenterCrop(image_size))
    
    return img_transforms
