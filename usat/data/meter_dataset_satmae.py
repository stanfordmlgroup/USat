import os
import random
from pathlib import Path

import geojson
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from usat.utils.builder import DATASET
from usat.utils.sentinel import (SatMAENormalize, MinMaxNormalize, 
                                     ConsistentRandomCrop, ConsistentRadomHorizontalFlip,
                                     ConsistentRadomVerticalFlip)


@DATASET.register_module()
class MeterMLSatMAEDataset(Dataset):

    STATS = {
        'sentinel-1': {
            'min': [0, 0],
            'max': [4095, 4095],
            'mean': [1653.2279, 1257.8794],
            'std': [290.1448, 280.2737],
            # minmax entries are computed after rescaling using the min & max
            'minmax_mean': [0.4018, 0.3052],
            'minmax_std': [0.0709, 0.0684],
            # 2std entries are computed after rescaling using the mean +- 2*std
            '2std_mean': [0.5018, 0.5013],
            '2std_std': [0.2220, 0.2355]
        },
        'sentinel-2-10m': {
            'min': [0, 0, 0, 0],
            'max': [16383, 16383, 16383, 16383],
            'mean': [1027.2854, 1139.7004, 1195.8230, 2635.9023],
            'std': [ 729.7542, 549.4569, 519.0011, 1053.8544],
            'minmax_mean': [0.0607, 0.0676, 0.0710, 0.1589],
            'minmax_std': [0.0446, 0.0336, 0.0317, 0.0643],
            '2std_mean': [0.4859, 0.4843, 0.4844, 0.4981],
            '2std_std': [0.2003, 0.1825, 0.1780, 0.2364]
        },
        'sentinel-2-20m': {
            'min': [0, 0, 0, 0, 0, 0],
            'max': [16383, 16383, 16383, 16383, 16383, 16383],
            'mean': [1270.1216, 2233.1870, 2722.1499, 2968.0166, 2103.8994, 1304.2429],
            'std': [ 666.4895, 798.9017, 1053.4434, 1164.7076, 997.3839, 861.1730],
            'minmax_mean': [0.0756, 0.1343, 0.1642, 0.1792, 0.1265, 0.0777],
            'minmax_std': [0.0407, 0.0488, 0.0643, 0.0711, 0.0609, 0.0526],
            '2std_mean': [0.4865, 0.4980, 0.4982, 0.4998, 0.4947, 0.4916],
            '2std_std': [0.2018, 0.2238, 0.2352, 0.2360, 0.2398, 0.2321]
        },
        'sentinel-2-60m': {
            'min': [0, 0, 0],
            'max': [11701, 10412, 6012],
            'mean': [1409.5294, 807.4094, 16.8547],
            'std': [413.6848, 419.1820, 55.8404],
            'minmax_mean': [0.1185, 0.0756, 0.0008],
            'minmax_std': [0.0354, 0.0403, 0.0093],
            '2std_mean': [0.4858, 0.4918, 0.4916],
            '2std_std': [0.1798, 0.2253, 0.0576]
        },
        'naip': {
            'mean': [0.4323, 0.4656, 0.3873, 0.4848],
            'std': [0.1865, 0.1578, 0.1506, 0.2104]
        }
    }
    
    PRODUCT_BAND_MAP = {
        'naip': {
            0: "NAIP:Red",
            1: "NAIP:Green",
            2: "NAIP:Blue",
            3: "NAIP:NIR"
        },
        "sentinel-1": {
            0: "S1:VV",
            1: "S1:VH"
        },
        "sentinel-2-10m": {
            0: "S2:Red",
            1: "S2:Green",
            2: "S2:Blue",
            3: "S2:NIR"
        },
        "sentinel-2-20m": {
            0: "S2:RE1",
            1: "S2:RE2",
            2: "S2:RE3",
            3: "S2:RE4",
            4: "S2:SWIR1",
            5: "S2:SWIR2",
        },
        "sentinel-2-60m": {
            0: "S2:CoastAerosal",
            1: "S2:WaterVapor",
            2: "S2:Cirrus"
        }
    }

    PRODUCT_BAND_MAP = {
        'naip': {
            0: "NAIP:Red",
            1: "NAIP:Green",
            2: "NAIP:Blue",
            3: "NAIP:NIR"
        },
        "sentinel-1": {
            0: "S1:VV",
            1: "S1:VH"
        },
        "sentinel-2-10m": {
            0: "S2:Red",
            1: "S2:Green",
            2: "S2:Blue",
            3: "S2:NIR"
        },
        "sentinel-2-20m": {
            0: "S2:RE1",
            1: "S2:RE2",
            2: "S2:RE3",
            3: "S2:RE4",
            4: "S2:SWIR1",
            5: "S2:SWIR2",
        },
        "sentinel-2-60m": {
            0: "S2:CoastAerosal",
            1: "S2:WaterVapor",
            2: "S2:Cirrus"
        }
    }

    ALL_BAND_NAMES = [name for _, bands in PRODUCT_BAND_MAP.items() for _, name in bands.items()]

    GSDS = {
        'naip': 1,
        'sentinel-1': 10,
        'sentinel-2-10m': 10,
        'sentinel-2-20m': 20,
        'sentinel-2-60m': 60,
    }

    LABEL_MAP = {
        'CAFOs': 0,
        'WWTreatment': 1,
        'RefineriesAndTerminals': 2,
        'Landfills': 3,
        'Mines': 4,
        'ProcessingPlants': 5
    }

    # This directory is missing Sentinel-1 and partial Sentinel-2 data
    SKIPPED_PATHS = ['train_images/train_images_3/38.8205231_-75.7927729']

    def __init__(self,
                 base_path: str,
                 split: str = 'train',
                 products: list = ['naip', 'sentinel-1', 'sentinel-2-10m', 'sentinel-2-20m', 'sentinel-2-60m'],
                 discard_bands: list = [],
                 standardize: bool = True,
                 full_return: bool = False,
                 img_size: int = 72,
                 data_percent: float = 1.0,
                 pad: bool = False,
                 custom_transform = None):

        # Verify that the products are valid
        assert all(product in ['naip', 'sentinel-1', 'sentinel-2-10m', 'sentinel-2-20m', 'sentinel-2-60m'] for product in products)
        assert all(band in self.ALL_BAND_NAMES for band in discard_bands)

        # Sort products from largest to smallest GSD so that our consistent 
        # random crop can utilize the same order
        largest_to_smallest_gsd_products = [product[0] for product in sorted(self.GSDS.items(), key=lambda x: x[1], reverse=True) if product[0] in products]

        self.base_path = base_path
        self.products = largest_to_smallest_gsd_products
        self.split = split
        self.standardize = standardize
        self.full_return = full_return
        self.img_size = img_size
        self.pad = pad
        self.data_percent = data_percent
        self.discard_bands = discard_bands

        self.custom_transform = self._build_transforms(split == "train", largest_to_smallest_gsd_products)
        self.metadata = self._load_metadata(split)
        self.paths = self._load_paths(split)

    def _load_metadata(self, split):
        refined_metadata = {}
        metadata_file = os.path.join(self.base_path, f'{split}_dataset.geojson')
        with open(metadata_file, 'r') as f:
            metadata = geojson.load(f)
        for feature in metadata['features']:
            folder_name = feature['properties']['Image_Folder']
            label = feature['properties']['Type']
            # negative & roundabout indicate absence of all classes
            class_idxs = [self.LABEL_MAP[cl.strip()] for cl in label.split("-") if cl.strip() not in ["Negative", "Roundabout"]]
            target = torch.zeros(len(self.LABEL_MAP), dtype=torch.int)
            target[class_idxs] = 1
            refined_metadata[folder_name] = target
        return refined_metadata

    def _load_paths(self, split):
        paths = []
        split_dir = f"{split}_images"
        split_dir_path = os.path.join(self.base_path, split_dir)
        if split == 'train':
            for sub_dir in ["train_images_1", "train_images_2", "train_images_3"]:
                train_img_path = os.path.join(self.base_path, split_dir_path, sub_dir)
                paths.extend([p for img_dir in os.listdir(train_img_path) if (p := os.path.join(split_dir, sub_dir, img_dir)) not in self.SKIPPED_PATHS])
        else:
            paths.extend([p for img_dir in os.listdir(split_dir_path) if (p := os.path.join(split_dir, img_dir)) not in self.SKIPPED_PATHS])
        paths = random.sample(paths, int(self.data_percent * len(paths)))
        return paths

    def _build_transforms(self, is_train, products):
        custom_transforms = {}

        # Assume the order is sorted from smallest to largest img_size b/c
        # we sorted in class init

        #img_sizes = [self.ground_cover / self.GSDS[product] for product in products]
        #cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(products))
        cons_vertical_flip = ConsistentRadomVerticalFlip(len(products))

        for idx, product in enumerate(products):
            t = []

            if product != "naip":
                t.append(SatMAENormalize(self.STATS[product]['mean'], self.STATS[product]['std'], scale=255))
                mean_key = "2std_mean"
                std_key = "2std_std"
                #t.append(MinMaxNormalize(self.STATS[product]['min'], self.STATS[product]['max'], scale=255))
                #mean_key = "minmax_mean"
                #std_key = "minmax_std"
            else:
                mean_key = "mean"
                std_key = "std"

            t.append(transforms.ToTensor()) 

            if self.standardize:
                t.append(transforms.Normalize(self.STATS[product][mean_key], self.STATS[product][std_key]))

            if product == "sentinel-2-20m":
                # Re-scale the image to be the same size
                interpol_mode = transforms.InterpolationMode.BICUBIC
                t.append(transforms.Resize(72, interpolation = interpol_mode))

            # At this point, all images should be the same size, scale up to img_size?

            if is_train:
                #t.append(cons_rand_crop)
                t.append(cons_horiz_flip)
                t.append(cons_vertical_flip)
            else:
                # For each grouped product, we will need a different input size
                #t.append(transforms.CenterCrop(img_sizes[idx]))
                t.append(transforms.CenterCrop(self.img_size))
            
            custom_transforms[product] = transforms.Compose(t)
        return custom_transforms

    def __len__(self):
        return len(self.paths)    

    def _load_img(self, path, product):
        if product == "naip":
            img_path = os.path.join(path, f"{product}.png")
            img_data = np.array(Image.open(img_path))
        else:
            img_path = os.path.join(path, f"{product}.npy")
            img_data = np.load(img_path).astype(np.int32)   # originally uint16, but pytorch can't convert
        return self.custom_transform[product](img_data)

    def __getitem__(self, idx):
        # The sentinel images appear to be TIFF loaded files, so they need
        # to be normalized to 0-255 (or 0-1)
        # naip.png -> 4 channel PNG w/ NIR in alpha channel
        # sentinel-1.npy -> 72, 72, 2 (VV, VH)
        # sentinel-2-10m.npy -> 72, 72, 4 (red, green, blue, NIR)
        # sentinel-2-20m.npy -> 36, 36, 6 (RE1, RE2, RE3, RE4, SWIR1, SWIR2)
        # sentinel-2-60m.npy -> 12, 12, 3 (coastal aerosol, water vapor, cirrus)
        imgs = {}
        img_path = self.paths[idx]
        full_path = os.path.join(self.base_path, img_path)

        # NOTE: the order that we iterate through products needs to match
        #       the order that we initialized ConsistentRandomTransform so
        #       the correct scale is applied to each image
        for product in self.products:
            # Break the product into its bands
            img = self._load_img(full_path, product)
            for idx, band_name in self.PRODUCT_BAND_MAP[product].items():
                if band_name not in self.discard_bands:
                    imgs[band_name] = img[idx, :, :][None, ...]
        
        if self.split == "train":
            split_path = img_path.split("/")[-3:]
            metadata_key = "/".join([split_path[0], split_path[-1]])
        else:
            metadata_key = "/".join(img_path.split("/")[-2:])
        label = self.metadata[metadata_key]
        if self.full_return:
            return imgs, label, idx
        return imgs, label

    def collate_fn(self, batch):
        if self.full_return:
            imgs, labels, paths = zip(*batch)
        else:
            imgs, labels = zip(*batch)

        collated_imgs = {}
        for product in self.products:
            for band_name in self.PRODUCT_BAND_MAP[product].values():
                if band_name not in self.discard_bands:
                    collated_imgs[band_name] = torch.stack([img[band_name] for img in imgs])
        labels = torch.stack(labels)

        if self.full_return:
            return collated_imgs, labels, paths
        return collated_imgs, labels
