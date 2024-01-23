from typing import Callable, Collection
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import os 

from usat.utils.builder import DATASET
from usat.utils.constants import NAIP_MEAN, NAIP_STD
from usat.utils.transform import RandomFlip, RandomRotate

@DATASET.register_module()
class NaipDataset(Dataset):
    def __init__(self,
        base_path: str,
        split: str =  'train',
        standardize: bool = False,
        bands: Collection[int]   = [0,1,2],
        custom_transform: Callable = None) -> None:
        """
        Keyword arguments:
            base_path: path to the metadata csv file
            split: [train, test]
            standardize: standardize data using train split mean and stdev
            bands: list of index of the bands to use. 
                          [0,1,2] for RGB or [0,1,2,3] for RGB+NIR. 
            custom_transform: a torchvision.transforms function applied to images. 
        """
        meta = pd.read_csv(base_path)
        meta = meta[meta.split_str == split]
        self.paths = [os.path.join(os.path.dirname(base_path),x) for x in meta['file_name']]
        self.labels = list(meta.y)
        self.bands = bands

        # order of the transformations follows SatMAE
        all_transforms = [T.ToTensor()]
        if custom_transform:
            all_transforms.append(custom_transform)
        if split == "train":
            all_transforms.append(RandomFlip(flip_ud = True, flip_lr = True))
            all_transforms.append(RandomRotate())
        if standardize:
            all_transforms.append(T.Normalize([NAIP_MEAN[i] for i in bands], [NAIP_STD[i] for i in bands]))
        all_transforms.append(T.Resize(224))
        self.transform_compose = T.Compose(all_transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        arr = np.load(path)
        img = self.transform_compose(Image.fromarray(arr[:,:,self.bands]))
        return (img, label)


@DATASET.register_module()
class NaipDatasetUsat(Dataset):
    bands_map = {0:'NAIP:Red', 1:'NAIP:Green', 2:'NAIP:Blue', 3:'NAIP:NIR'}

    def __init__(self,
        base_path: str,
        split: str =  'train',
        standardize: bool = False,
        bands: Collection[int]   = [0,1,2],
        ground_cover: int = 80,
        custom_transform: Callable = None) -> None:
        """
        Keyword arguments:
            base_path: path to the metadata csv file
            split: [train, test]
            standardize: standardize data using train split mean and stdev
            bands: list of index of the bands to use. 
                          [0,1,2] for RGB or [0,1,2,3] for RGB+NIR. 
            custom_transform: a torchvision.transforms function applied to images. 
        """
        meta = pd.read_csv(base_path)
        # val and test are both considered val
        if split == 'train':
            meta = meta[meta.split_str == 'train']
        else:
            meta = meta[meta.split_str != 'train']
        self.paths = [os.path.join(os.path.dirname(base_path),x) for x in meta['file_name']]
        self.labels = list(meta.y)
        self.bands = bands

        # order of the transformations follows SatMAE
        all_transforms = [T.ToTensor()]
        if custom_transform:
            all_transforms.append(custom_transform)
        if standardize:
            all_transforms.append(T.Normalize([NAIP_MEAN[i] for i in bands], [NAIP_STD[i] for i in bands]))
        # resize to gsd = 1
        all_transforms.append(T.Resize(60))
        # Crop or pad to ground cover
        all_transforms.append(T.CenterCrop(ground_cover))
        if split == "train":
            all_transforms.append(RandomFlip(flip_ud = True, flip_lr = True))
        self.transform_compose = T.Compose(all_transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        arr = np.load(path)
        # img = self.transform_compose(Image.fromarray(arr[:,:,self.bands]))
        img = self.transform_compose(Image.fromarray(arr[:,:,self.bands]))
        items = {}
        for band_idx in self.bands:
            items[self.bands_map[band_idx]] = img[band_idx][None,:,:]

        return items, label


    def collate_fn (self, batch):
        imgs, labels = zip(*batch)
        imgs_batch = {}
        for product_band in imgs[0].keys():
            imgs_batch[product_band] = torch.stack([img[product_band] for img in imgs]) # bs,1,H,W
        labels = torch.tensor(labels)
        return imgs_batch, labels

if __name__ == '__main__':
    rgb_dataset = NaipDatasetUsat(split = 'train', bands = [0,1,2],standardize = True,base_path = '/scr/naip/land_cover_representation/metadata_filtered.csv')
    print("3 channel train: ")
    print(rgb_dataset[0][0]['NAIP:Red'].shape)
    rgbn_dataset = NaipDatasetUsat(split = 'test', bands = [0,1,2,3], standardize = True,base_path = '/scr/naip/land_cover_representation/metadata_filtered.csv')
    print("4 channel val: ")
    print(rgbn_dataset[0][0]['NAIP:NIR'].shape)