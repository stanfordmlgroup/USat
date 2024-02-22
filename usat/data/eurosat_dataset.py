import os
import ssl
from typing import Callable, Collection

import numpy as np
import torch
from torchgeo.datasets  import EuroSAT
import torchvision
import torchvision.transforms as T 

from usat.utils.builder import DATASET 
from usat.utils.constants import EUROSAT_MEANS, EUROSAT_STDS, EUROSAT_SCALED_MEANS, EUROSAT_SCALED_STSDS
from usat.utils.sentinel import (SentinelNormalize, ConsistentRadomHorizontalFlip,
                                 ConsistentRadomVerticalFlip, ConsistentRandomCrop)

# NOTE: This is needed to download the EuroSAT dataset via torchgeo
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


@DATASET.register_module()
class EuroSatDatasetUSat(EuroSAT):
    """ Dataloader for the EuroSAT dataset, leveraging the torchgeo EuroSAT
    implementation: https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat
    """ 

    BAND_MAP = {
        "B01": "S2:CoastalAerosal",
        "B05": "S2:RE1",
        "B06": "S2:RE2",
        "B07": "S2:RE3",
        "B08A": "S2:RE4",
        "B11": "S2:SWIR1",
        "B12": "S2:SWIR2",
        "B02": "S2:Blue",
        "B03": "S2:Green",
        "B04": "S2:Red",
        "B08": "S2:NIR",
        "B09": "S2:WaterVapour",
        "B10": "S2:Cirrus",
    }

    GSD_MAP = {
        "B01": 60,
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B05": 20,
        "B06": 20,
        "B07": 20,
        "B08": 10,
        "B08A": 20,
        "B09": 60,
        "B10": 60,
        "B11": 20,
        "B12": 20,
    } 

    def __init__(self, 
                 base_path: str,
                 split: str =  'train',
                 standardize: bool = True,
                 ground_cover: int = 640,
                 rescale_img: bool = False,
                 image_size: int = 224,
                 data_percent: float = 1.0,
                 bands: Collection[str] = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"],
                 discard_bands: Collection[str] = ["B01", "B09", "B10"],
                 custom_transform: Callable = None,
                 download: bool = False):
        """
        base_path: path to data 
        split: 'train', 'val', or 'test'
        standardize: apply normalization based on per-band mean and std
        ground_cover: square meter ground coverage of image
        rescale_img: rescale the image or not
        image_size: if 'rescale_img' is true, rescale to this size, otherwise ignore
        data_percent: percent of data in the split to use
        bands: spectral bands to use - e.g. ['NAIP:Red', 'S2:Red']
        discard_bands: bands to discard - e.g. ['NAIP:Red', 'S2:Red']
        custom_transform: not used
        download: if True, will download the dataset to 'base_path'
        """
        super().__init__(
            root=base_path,
            split=split,
            bands=bands,
            transforms=torchvision.transforms.Lambda(self.__transform_wrapper__),
            download=download
        )

        # NOTE: if you want to use a data_percent other than 1.0, first run the
        # 'create_partial_dataset' function to create a new split file
        if data_percent != 1.0 and split == 'train':
            valid_fns = set()
            with open(os.path.join(base_path, f"eurosat-{split}_{data_percent}.txt")) as f:
                for fn in f:
                    valid_fns.add(fn.strip().replace(".jpg", ".tif"))
            is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns
            _, class_to_idx = self.find_classes(self.root)
            self.samples = self.make_dataset(self.root, class_to_idx, self.extensions, is_in_split)
            self.imgs = self.samples
            self.targets = [s[1] for s in self.samples]

        self.standardize = standardize
        self.rescale_img = rescale_img
        self.image_size = image_size
        self.band_names = [self.BAND_MAP[b] for b in bands]
        self.band_gsds = [self.GSD_MAP[b] for b in bands]
        self.ground_cover = ground_cover
        self.discard_bands = discard_bands
        self.custom_transform = self._build_transforms(split == 'train')

    def _build_transforms(self, is_train):
        custom_transforms = {}
        # Don't use 60m bands
        reordered_band_idxs = [self.all_band_names.index(b) for b in self.BAND_MAP if b not in self.discard_bands]
        if self.rescale_img:
            img_sizes = [self.image_size for _ in reordered_band_idxs]
        else:
            img_sizes = [self.ground_cover // self.GSD_MAP[self.all_band_names[idx]] for idx in reordered_band_idxs]

        cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(img_sizes))
        cons_vert_flip = ConsistentRadomVerticalFlip(len(img_sizes))
        for order, idx in enumerate(reordered_band_idxs):
            img_transform = []
            img_transform.append(SentinelNormalize(EUROSAT_MEANS[idx], EUROSAT_STDS[idx], 1.0))

            if self.standardize:
                img_transform.append(T.Normalize(EUROSAT_SCALED_MEANS[idx], EUROSAT_SCALED_STSDS[idx]))

            # EuroSAT images are scaled to fixed size, resize back to
            # orig dimensions based on GSD, applicable for 20m and 60m
            img_transform.append(T.Resize(img_sizes[order], interpolation=T.InterpolationMode.BICUBIC))

            if is_train:
                img_transform.append(cons_rand_crop)
                img_transform.append(cons_horiz_flip)
                img_transform.append(cons_vert_flip)
            else:
                img_transform.append(T.CenterCrop(img_sizes[order]))
        
            custom_transforms[self.BAND_MAP[self.all_band_names[idx]]] = T.Compose(img_transform)
        return custom_transforms

    def __transform_wrapper__(self, object):   
        """ Wrapper transform to pack and unpack from dictionary while filtering
        and applying transforms on each band
        """
        image, label = object['image'], object['label'] 
        reordered_band_idxs = [self.all_band_names.index(b) for b in self.BAND_MAP if b not in self.discard_bands]
        imgs = {}
        for idx in reordered_band_idxs:
            band_name = self.BAND_MAP[self.all_band_names[idx]]
            if self.all_band_names[idx] not in self.discard_bands:
                imgs[band_name] = self.custom_transform[band_name](image[idx, :, :][None, :, :])
        return (imgs, label) 

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        collated_imgs = {}
        for band in imgs[0].keys():
            collated_imgs[band] = torch.stack([img[band] for img in imgs])
        labels = torch.stack(labels)
        return collated_imgs, labels


def create_partial_dataset(root, split, data_percent=1.0):
    import os
    with open(os.path.join(root, f"eurosat-{split}.txt")) as f:
        files  = f.readlines()
    rng = np.random.default_rng(42)
    indices = rng.choice(range(len(files)), int(data_percent * len(files)))
    final_files = np.array(files)[indices].tolist()
    with open(os.path.join(root, f"eurosat-{split}_{data_percent}.txt"), 'w') as f:
        f.writelines(final_files)
    print(f"Total examples in {data_percent} {split}: {len(final_files)}")


if __name__ == '__main__':
    train_eurosat_dataset = EuroSatDatasetUSat(base_path= '/scr/eurosat', standardize=True, use_satmae=True, image_size=96, download=True, data_percent=1.0)
    print(len(train_eurosat_dataset))

    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.0062)
    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.06173)
    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.617284)