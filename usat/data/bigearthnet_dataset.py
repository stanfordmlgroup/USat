import os
import re
import ssl
from typing import Tuple, Dict, List, Callable

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import BigEarthNet
import torchvision.transforms as T

from usat.utils.builder import DATASET
from usat.utils.sentinel import (SentinelNormalize, ConsistentRadomHorizontalFlip,
                                 ConsistentRadomVerticalFlip, ConsistentRandomCrop)
from usat.utils.constants import (BEN_TF_NO_BAD_PATCH_TRAIN_MEAN,
                                  BEN_TF_NO_BAD_PATCH_TRAIN_STD,
                                  BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN,
                                  BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD,
                                  SENTINEL_2_BANDS)


# NOTE: This is needed to download the BEN dataset via torchgeo due to SSL cert issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


@DATASET.register_module()
class BigEarthNetDatasetUSat(BigEarthNet):
    """ Dataloader for the BigEarthNet dataset, leveraging the torchgeo BigEarthNet
    implementation: https://torchgeo.readthedocs.io/en/latest/api/datasets.html#bigearthnet
    """ 

    # NOTE: make sure you are using torchgeo version >= 0.4.1 with label
    # mapping fix for BigEarthNet

    splits_metadata = {
        # torchgeo splits are different than splits used by prior research papers, so
        # we opt to use the splits from tensorflow (which are the same)
        "torchgeo": {
            "train": {
                "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv?inline=false",  # noqa: E501
                "filename": "bigearthnet-tg-train.csv",
                "md5": "623e501b38ab7b12fe44f0083c00986d",
            },
            "val": {
                "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv?inline=false",  # noqa: E501
                "filename": "bigearthnet-tg-val.csv",
                "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
            },
            "test": {
                "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv?inline=false",  # noqa: E501
                "filename": "bigearthnet-tg-test.csv",
                "md5": "697fb90677e30571b9ac7699b7e5b432",
            },
        },
        "tensorflow": {
            "train": {
                "url": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt",
                "filename": "bigearthnet-tf-train.csv",
                "md5": "343f7224951da0879cabcceae7b34a64"
            },
            "val": {
                "url": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt",
                "filename": "bigearthnet-tf-val.csv",
                "md5": "8fe0292aa346c0833850a322e1c1e783"
            },
            "test": {
                "url": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt",
                "filename": "bigearthnet-tf-test.csv",
                "md5": "efce8d370cf5f55afc415ace88605218"
            },
        }
    }

    bad_patches_metadata = {
        "snow": {
            "url": "http://bigearth.net/static/documents/patches_with_seasonal_snow.csv",
            "filename": "patches_with_seasonal_snow.csv",
            "md5": "7388ec3bee2d519846f01ecb9f95cc4d"
        },
        "cloud": {
            "url": "http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv",
            "filename": "patches_with_cloud_and_shadow.csv",
            "md5": "7388ec3bee2d519846f01ecb9f95cc4d"
        }
    }

    # Ordered by lowest to highest resolution
    band_map = {
        "B01": "S2:CoastalAerosal",
        "B09": "S2:WaterVapour",
        #"B10": "S2:Cirrus",        # BEN doesn't have B10
        "B05": "S2:RE1",
        "B06": "S2:RE2",
        "B07": "S2:RE3",
        "B8A": "S2:RE4",
        "B11": "S2:SWIR1",
        "B12": "S2:SWIR2",
        "B02": "S2:Blue",
        "B03": "S2:Green",
        "B04": "S2:Red",
        "B08": "S2:NIR",
    }

    gsd_map = {
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B05": 20,
        "B06": 20,
        "B07": 20,
        "B08": 10,
        "B8A": 20,
        "B11": 20,
        "B12": 20
    }
    
    data_dirname = "bigearthnet"


    def __init__(self, 
                 base_path: str,
                 split: str = 'train',
                 standardize: bool = False,
                 ground_cover: int = 720,
                 rescale_img: bool = False,
                 image_size: int = 224,
                 data_percent: float = 0.1,
                 # B10 not present in BEN, discard B01 and B09 b/c 60m
                 bands: List[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                 random_seed: int = 42,
                 num_classes: int = 19,
                 exclude_bad_patches = True,
                 checksum: bool = False,
                 custom_transform: Callable = None,
                 download: bool = False):
        """Constructor for BigEarthNetDataset.

        base_path: path to data 
        split: 'train', 'val', or 'test'
        standardize: apply normalization based on per-band mean and std
        ground_cover: square meter ground coverage of image
        rescale_img: rescale the image or not
        image_size: if 'rescale_img' is true, rescale to this size, otherwise ignore
        data_percent: percent of data in the split to use
        bands: spectral bands to use
        random_seed: random seed used for subsetting dataset
        num_classes: number of BigEarthNet class labels to use - 19 or 43
        exclude_bad_patches: whether or not to exclude cloud and snow
        checksum: if True, checks MD5 of downloaded files
        custom_transform: not used
        download: if True, downloads dataset and stores at base_path
        """

        if bands is not None and 'B10' in bands:
            raise Exception("B10 not supported for BigEarthNet")

        # Class variables inherited from parent
        assert num_classes in [19, 43]
        self.split = 'val' if split == 'eval' else split
        self.root = base_path
        self.bands = 's2'
        self.num_classes = num_classes
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}

        BEN_BANDS = SENTINEL_2_BANDS.copy()
        BEN_BANDS.remove('B10') # BEN doesn't have B10 to begin with

        self.random_seed = random_seed
        self.ground_cover = ground_cover
        self.data_percent = data_percent
        self.rescale_img = rescale_img
        self.standardize = standardize
        self.splits_metadata = self.splits_metadata['tensorflow']
        self.image_size = image_size
        self.band_regex = re.compile(".*_(.*)\.tif")
        self.ben_bands = bands
        # Get the correct idx for normalization
        self.stat_idxs = [idx for idx, band in enumerate(BEN_BANDS) if band in bands]
        # Setup regex to exclude all omitted bands
        self.exclude_re = None
        if bands:
            exclude_bands =  list(set(BEN_BANDS) - set(bands))
            self.exclude_re = re.compile(f".*({'|'.join(exclude_bands)})\.tif$")

        self.mean = [BEN_TF_NO_BAD_PATCH_TRAIN_MEAN[ind] for ind in self.stat_idxs]
        self.std = [BEN_TF_NO_BAD_PATCH_TRAIN_STD[ind] for ind in self.stat_idxs]
        self.scaled_mean = [BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN[ind] for ind in self.stat_idxs]
        self.scaled_std = [BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD[ind] for ind in self.stat_idxs]

        self._verify()
        self.bad_patches = self._load_bad_patches(exclude_bad_patches)
        self.folders = self._load_folders()
        self.custom_transform = self._build_transforms(split == 'train')
 
    def _build_transforms(self, is_train):
        custom_transforms = {}
        reduced_map = [band for band in self.band_map if band in self.ben_bands]
        if self.rescale_img:
            img_sizes = [self.image_size for _ in reduced_map]
        else:
            img_sizes = [self.ground_cover / self.gsd_map[band] for band in reduced_map]

        cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(self.stat_idxs))
        cons_vert_flip = ConsistentRadomVerticalFlip(len(self.stat_idxs))
        for idx, band in enumerate(reduced_map):
            img_transforms = []
            img_transforms.append(SentinelNormalize(self.mean[idx], self.std[idx]))
            img_transforms.append(T.ToTensor())

            if self.standardize:
                img_transforms.append(T.Normalize(self.scaled_mean[idx], self.scaled_std[idx]))

            if is_train:
                img_transforms.append(cons_rand_crop)
                img_transforms.append(cons_horiz_flip)
                img_transforms.append(cons_vert_flip)
            else:
                img_transforms.append(T.CenterCrop(img_sizes[idx]))
            
            custom_transforms[band] = T.Compose(img_transforms)
        return custom_transforms

    def _load_bad_patches(self, exclude_bad_patches: bool) -> set[str]:
        """Loads the set of patches to be ignored

        Keyword Arguments:
            exclude_bad_patches: whether or not cloudy and snowy patches should be excluded
        """
        bad_patches = set()
        if exclude_bad_patches:
            cloud_patches = os.path.join(self.root, self.bad_patches_metadata["cloud"]["filename"])
            snow_patches = os.path.join(self.root, self.bad_patches_metadata["snow"]["filename"])
            with open(cloud_patches, 'r') as f:
                bad_patches.update(f.read().splitlines())
            with open(snow_patches, 'r') as f:
                bad_patches.update(f.read().splitlines())
        return bad_patches

    def _load_image(self, index: int) -> Tensor:
        """Override _load_image to apply custom transform at load stage before converted to a Torch tensor.
        """
        paths, band_order = self._load_paths(index)
        # Save file paths based on saved data
        imgs = {}
        # Need to apply fixed_img_transforms in the same order as band_map
        reduced_map = [band for band in self.band_map if band in self.ben_bands]
        for band_name in reduced_map:
            idx = band_order.index(band_name)
            path = paths[idx]
            # Don't resample until after loading is done
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_dtype="int32",
                )
                imgs[self.band_map[band_name]] = self.custom_transform[band_name](array)
        return imgs

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Override __getitem__ to apply transform only to images and not labels.
        """
        image = self._load_image(index)
        label = self._load_target(index)
        return image, label

    def _load_paths(self, index: int) -> List[str]:
        """Override _load_paths to drop unused bands by omitting them from indexed paths.
        """
        paths = super()._load_paths(index)
        filtered_paths, all_band_order = [], []
        for path in paths:
            if self.exclude_re and self.exclude_re.match(path):
                continue
            filtered_paths.append(path)
            all_band_order.append(self.band_regex.match(path).group(1))
        return filtered_paths, all_band_order

    def _load_folders(self) -> List[Dict[str, str]]:
        """Override _load_folders to perform dataset reduction during indexing rather than runtime.
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
        folders = [
            {
                "s2": os.path.join(self.root, dir_s2, line)
            }
            for line in lines if line not in self.bad_patches
        ]

        if self.split in ['train', 'val']:
            # Random sampling function borrowed from SeCo
            # https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/utils/data.py#L26-L29
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(range(len(folders)), int(self.data_percent * len(folders)))
            folders = np.array(folders)[indices].tolist()
        return folders

    def _verify(self) -> None:
        """ Override _verify to ensure bad patch metadata files are also downloaded.
        """
        super()._verify()
        # Verify that bad patch CSVs are also downloaded
        urls = [self.bad_patches_metadata[k]["url"] for k in self.bad_patches_metadata]
        md5s = [self.bad_patches_metadata[k]["md5"] for k in self.bad_patches_metadata]
        filenames = [self.bad_patches_metadata[k]["filename"] for k in self.bad_patches_metadata]
        if all([os.path.exists(os.path.join(self.root, filename)) for filename in filenames]):
            return
        # Download files
        for url, filename, md5 in zip(urls, filenames, md5s):
            self._download(url, filename, md5)    

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        collated_imgs = {}
        band_map_vals = [band_name for band, band_name in self.band_map.items() if band in self.ben_bands]
        for band in band_map_vals:
            collated_imgs[band] = torch.stack([img[band] for img in imgs])
        labels = torch.stack(labels)
        return collated_imgs, labels


if __name__ == '__main__':
    train_ben_dataset = BigEarthNetDatasetUSat(split='train', base_path='/scr/bigearthnet', 
                                               standardize=True, download=False, ground_cover=640, data_percent=0.1)
    print(len(train_ben_dataset))
    print(train_ben_dataset[0][0]['S2:Red'].shape)
    print(train_ben_dataset[0][0]['S2:SWIR1'].shape)
