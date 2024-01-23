import math
import os
import random
import re
import ssl
from typing import Optional, Callable, Tuple, Dict, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torch import Tensor
from torchgeo.datasets import BigEarthNet
import torchvision.transforms as tv_transform

from usat.utils.builder import DATASET
from usat.utils.sentinel import (get_satmae_transforms, SatMAENormalize,
                                     ConsistentRadomHorizontalFlip, ConsistentRadomVerticalFlip,
                                     ConsistentRandomCrop)
from usat.utils.constants import (GENERAL_DATA_DIR, SENTINEL_2_BANDS,
                                      SENTINEL_1_BANDS,
                                      BEN_S1_TF_TRAIN_MEAN,
                                      BEN_S1_TF_TRAIN_STD,
                                      BEN_S1_TF_SCALED_TRAIN_MEAN,
                                      BEN_S1_TF_SCALED_TRAIN_STD,
                                      BEN_TG_TRAIN_MEAN, BEN_TG_TRAIN_STD,
                                      BEN_TG_SCALED_TRAIN_MEAN,
                                      BEN_TG_SCALED_TRAIN_STD,
                                      BEN_TF_NO_BAD_PATCH_TRAIN_MEAN,
                                      BEN_TF_NO_BAD_PATCH_TRAIN_STD,
                                      BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN,
                                      BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD)


@DATASET.register_module(overwrite=True)
class BigEarthNetDataset(BigEarthNet):
    """Wrapper for the BigEarthNet dataset. Additional parameters
    are added to the standard TorchGeo/Pytorch Dataset class that allow 
    for data filtering and augmentation choices that mirror those made
    in the SatMAE paper.
     - Only Sentinel-2 data is used
     - 10% subset of train is used
     - B1, B9, and B10 are dropped (B10 already omitted from BigEarthNet)

    Note: Number of examples on torchgeo (269695 train, 123723 val, 125866 test) 
    differs than numbers mentioned in SatMAE (354196 train, 118065 validation).
    This could be due to torchgeo filtering out fully covered images due to
    seasonal snow, cloud, and cloud shadow (BigEarthNet paper filters these).

    NOTE: When running this file directly, you might see a warning saying
    KeyError: 'BigEarthNetDataset is already registered in dataset'. This is due
    to the __init__.py file also importing the class, reulting in a duplicate
    registering. This is not an issue when training, but can be worked around
    by passing in "overwrite=True" as a param to the register_module function.

    TODO: Validate that the dataset and data filtering used matches the SatMAE
    paper once their code is made public.
    """

    splits_metadata = {
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

    data_dirname = "bigearthnet"


    def __init__(self, 
                 base_path: str = GENERAL_DATA_DIR,
                 split: str = 'train',
                 standardize: bool = False,
                 bands: List[str] = None,
                 custom_transform: Optional[Callable[[Tuple[Tensor, Tensor]],
                                      Tuple[Tensor, Tensor]]] = None,
                 sentinel_source: str = 's2',
                 num_classes: int = 19,
                 image_size: int = 224,
                 exclude_bad_patches = True,
                 splits_to_use: str = 'tensorflow',
                 download: bool = False,
                 checksum: bool = False,
                 satmae_transform: bool = True) -> None:
        """Constructor for BigEarthNetDataset.

        Keyword arguments:
            base_path: root directory for dataset
            split: train/val/test split to use
            standardize: normalize to 0 mean and 1 std
            bands: list of bands to use (if not specified, all
                   sentinel_source bands will be used)
            custom_transform: function/transform to apply to sample image
                              and label
            sentinel_source: use Sentinel-1/2 bands {s1, s2, all}
            num_classes: number of BigEarthNet classes to use {19, 43}
            image_size: size to scale image
            exclude_bad_patches: whether or not to exclude cloud and snow
            splits_to_use: tensorflow or pytorch train, val, split sets
            download: if True, downloads dataset and stores in root
            checksum: if True, checks MD5 of downloaded files
            satmae_transform: if True, apply SatMAE transforms
        """

        if bands is not None and 'B10' in bands:
            raise Exception("B10 not supported for BigEarthNet")
        if base_path == GENERAL_DATA_DIR:
            base_path = os.path.join(base_path, self.data_dirname)
        split = 'val' if split == 'eval' else split

        assert splits_to_use in ["torchgeo", "tensorflow"]
        assert split in self.splits_metadata[splits_to_use]
        assert sentinel_source in ["s1", "s2", "all"]
        assert num_classes in [19, 43]
        self.root = base_path
        self.split = split
        self.bands = sentinel_source
        self.num_classes = num_classes
        self.transforms = custom_transform
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}

        # Newly added to support 
        BEN_BANDS = SENTINEL_2_BANDS.copy()
        BEN_BANDS.remove('B10')
        stat_idxs = list(range(len(BEN_BANDS)))
        self.exclude_re = None
        self.standardize = standardize
        self.splits_to_use = splits_to_use
        self.splits_metadata = self.splits_metadata[splits_to_use]
        self.image_size = image_size

        if bands:
            assert all(band in BEN_BANDS for band in bands)
            exclude_bands =  list(set(BEN_BANDS) - set(bands))
            self.exclude_re = re.compile(f".*({'|'.join(exclude_bands)})\.tif$")
            stat_idxs = [ind for ind, val in enumerate(BEN_BANDS)
                                if val in bands] 

        # Reshape dims to match HxWxC b/c normalize before converting to tensor
        if self.splits_to_use == "torchgeo":
            self.mean = np.array([BEN_TG_TRAIN_MEAN[ind]
                                        for ind in stat_idxs]).reshape(1, 1, -1)
            self.std = np.array([BEN_TG_TRAIN_STD[ind]
                                        for ind in stat_idxs]).reshape(1, 1, -1)
            self.scaled_mean = np.array([BEN_TG_SCALED_TRAIN_MEAN[ind]
                                        for ind in stat_idxs])
            self.scaled_std = np.array([BEN_TG_SCALED_TRAIN_STD[ind]
                                        for ind in stat_idxs])
        elif self.splits_to_use == "tensorflow":
            self.mean = np.array([BEN_TF_NO_BAD_PATCH_TRAIN_MEAN[ind]
                                        for ind in stat_idxs]).reshape(1, 1, -1)
            self.std = np.array([BEN_TF_NO_BAD_PATCH_TRAIN_STD[ind]
                                        for ind in stat_idxs]).reshape(1, 1, -1)
            self.scaled_mean = np.array([BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN[ind]
                                        for ind in stat_idxs])
            self.scaled_std = np.array([BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD[ind]
                                        for ind in stat_idxs])
        else:
            raise Exception("Only 'torchgeo' and 'tensorflow' are valid split options")


        self.fixed_img_transforms = self._get_transform(satmae_transform, standardize)

        # BigEarthNet dataset source seems to have no (or an expired) 
        # SSL certificate causing download issues. This line can be removed
        # once it is fixed (also ssl import can be removed)
        ssl._create_default_https_context = ssl._create_unverified_context

        self._verify()
        self.bad_patches = self._load_bad_patches(exclude_bad_patches)
        self.folders = self._load_folders()
 
    def _get_transform(self, satmae_transform: bool, standardize: bool) -> Callable:
        """
        Creates a callable comprised of image transforms specified in SatMAE
        paper
        """
        img_transforms = []
        img_transforms.append(SatMAENormalize(self.mean, self.std))
        img_transforms.append(tv_transform.ToTensor())

        if satmae_transform:
            img_transforms.extend(get_satmae_transforms(self.split, self.image_size))

        if standardize:
            img_transforms.append(tv_transform.Normalize(self.scaled_mean, self.scaled_std))
        
        return tv_transform.Compose(img_transforms)

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
        """Override _load_image to apply custom transform at load
        stage before converted to a Torch tensor.
        
        Keyword arguments:
            index: index to return
        Returns:
            tensor: the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Don't resample until after loading is done
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=(self.image_size, self.image_size),
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        arrays = arrays.transpose(1, 2, 0).astype(np.float32)
        tensor = self.fixed_img_transforms(arrays)
        return tensor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Override __getitem__ to apply transform only to images and not
        labels.

        Keyword arguemnts:
            index: index to return
        Returns:
            (image, label): image and corresponding label for given index
        """
        item = super().__getitem__(index)
        return (item["image"].float(), item["label"])

    def _load_paths(self, index: int) -> List[str]:
        """Override _load_paths to drop unused bands by
        omitting them from indexed paths.

        Keyword arguments:
            index: index to return
        Returns:
            paths: list of file paths
        """
        paths = super()._load_paths(index)
        # Drop bands that are not needed
        if self.exclude_re:
            paths = [path for path in paths if not self.exclude_re.match(path)]
        return paths

    def _load_folders(self) -> List[Dict[str, str]]:
        """Override _load_folders to perform dataset
        reduction during indexing rather than runtime.

        Returns:
            folders: List[Dict[str, str]] - list of folders
        """

        # TorchGeo dataset has s1 and s2 data in the 
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        # If the CSV file being used is from TensorFlow, it 
        # won't have both S1 and S2 data, assume its only S2 data
        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        if self.splits_to_use == "tensorflow":
            folders = [
                {
                    "s2": os.path.join(self.root, dir_s2, pair[0])
                }
                for pair in pairs if pair[0] not in self.bad_patches
            ]
        # TorchGeo seems to already filter out all the bad patches
        elif self.splits_to_use == "torchgeo":
            folders = [
                {
                    "s1": os.path.join(self.root, dir_s1, pair[1]),
                    "s2": os.path.join(self.root, dir_s2, pair[0]),
                }
                for pair in pairs if pair
            ]

        if self.split in ['train', 'val']:
            # SatMAE uses 10% of train dataset
            # Random sampling function written to match SeCo
            # TODO: find a way to dynamically get seed into dataset
            # b/c default_rng call doesn't follow from global seeding
            rng = np.random.default_rng(42)
            indices = rng.choice(range(len(folders)), int(0.1 * len(folders)))
            folders = np.array(folders)[indices].tolist()
        return folders

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
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


@DATASET.register_module()
class BigEarthNetDatasetUSat(BigEarthNet):

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean"
        ],
    }

    splits_metadata = {
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
                "filename": "bigearthnet-tf-train-mod.csv",
                "md5": "343f7224951da0879cabcceae7b34a64"
            },
            "val": {
                "url": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt",
                "filename": "bigearthnet-tf-val-mod.csv",
                "md5": "8fe0292aa346c0833850a322e1c1e783"
            },
            "test": {
                "url": "https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt",
                "filename": "bigearthnet-tf-test-mod.csv",
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

    band_map = {
        #"B01": "S2:CoastalAerosal",
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
        #"B09": "S2:WaterVapour",
        #"B10": "S2:Cirrus",        # BEN doesn't have B10
        "VV": "S1:VV",
        "VH": "S1:VH",
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
        "B12": 20,
        "VV": 10,
        "VH": 10,

    }
    
    data_dirname = "bigearthnet"


    def __init__(self, 
                 base_path: str = GENERAL_DATA_DIR,
                 split: str = 'train',
                 standardize: bool = False,
                 bands: List[str] = None,
                 custom_transform: Optional[Callable[[Tuple[Tensor, Tensor]],
                                      Tuple[Tensor, Tensor]]] = None,
                 sentinel_source: str = 's2',
                 num_classes: int = 19,
                 image_size: int = 224,
                 ground_cover: int = 720,
                 exclude_bad_patches = True,
                 splits_to_use: str = 'tensorflow',
                 download: bool = False,
                 checksum: bool = False,
                 satmae_transform: bool = False,
                 use_satmae: bool = False,
                 data_percent: float = 0.1) -> None:
        """Constructor for BigEarthNetDataset.

        Keyword arguments:
            base_path: root directory for dataset
            split: train/val/test split to use
            standardize: normalize to 0 mean and 1 std
            bands: list of bands to use (if not specified, all
                   sentinel_source bands will be used)
            custom_transform: function/transform to apply to sample image
                              and label
            sentinel_source: use Sentinel-1/2 bands {s1, s2, all}
            num_classes: number of BigEarthNet classes to use {19, 43}
            image_size: size to scale image
            exclude_bad_patches: whether or not to exclude cloud and snow
            splits_to_use: tensorflow or pytorch train, val, split sets
            download: if True, downloads dataset and stores in root
            checksum: if True, checks MD5 of downloaded files
            satmae_transform: if True, apply SatMAE transforms
        """

        if bands is not None and 'B10' in bands:
            raise Exception("B10 not supported for BigEarthNet")
        if base_path == GENERAL_DATA_DIR:
            base_path = os.path.join(base_path, self.data_dirname)
        split = 'val' if split == 'eval' else split

        assert splits_to_use in ["torchgeo", "tensorflow"]
        assert split in self.splits_metadata[splits_to_use]
        assert sentinel_source in ["s1", "s2", "all"]
        assert num_classes in [19, 43]
        self.root = base_path
        self.split = split
        self.bands = sentinel_source
        self.num_classes = num_classes
        self.transforms = custom_transform
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}

        # Newly added to support 
        if sentinel_source == "all":
            # Note that SENTINEL_1 bands are added first per
            # the torchgeo implementation when sorting paths
            BEN_BANDS = SENTINEL_1_BANDS.copy()
            BEN_BANDS += SENTINEL_2_BANDS.copy()
            BEN_BANDS.remove('B10')

            BEN_MEAN = BEN_S1_TF_TRAIN_MEAN + BEN_TF_NO_BAD_PATCH_TRAIN_MEAN
            BEN_STD = BEN_S1_TF_TRAIN_STD + BEN_TF_NO_BAD_PATCH_TRAIN_STD
            BEN_SCALED_MEAN = BEN_S1_TF_SCALED_TRAIN_MEAN + BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN
            BEN_SCALED_STD = BEN_S1_TF_SCALED_TRAIN_STD + BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD

        elif sentinel_source == "s1":
            BEN_BANDS = SENTINEL_1_BANDS.copy()

            BEN_MEAN = BEN_S1_TF_TRAIN_MEAN
            BEN_STD = BEN_S1_TF_TRAIN_STD
            BEN_SCALED_MEAN = BEN_S1_TF_SCALED_TRAIN_MEAN
            BEN_SCALED_STD = BEN_S1_TF_SCALED_TRAIN_STD

        elif sentinel_source == "s2":
            BEN_BANDS = SENTINEL_2_BANDS.copy()
            BEN_BANDS.remove('B10')

            BEN_MEAN = BEN_TF_NO_BAD_PATCH_TRAIN_MEAN
            BEN_STD = BEN_TF_NO_BAD_PATCH_TRAIN_STD
            BEN_SCALED_MEAN = BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN
            BEN_SCALED_STD = BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD

        #self.stat_idxs = list(range(len(BEN_BANDS)))
        self.stat_idxs = [idx for idx, band in enumerate(BEN_BANDS) if band not in ['B10', 'B01', 'B09']]
        try:
            BEN_BANDS.remove('B01')
            BEN_BANDS.remove('B09')
        except:
            pass
        self.exclude_re = None
        self.standardize = standardize
        self.splits_to_use = splits_to_use
        self.splits_metadata = self.splits_metadata[splits_to_use]
        self.image_size = image_size
        self.band_regex = re.compile(".*_(.*)\.tif")
        self.ben_bands = BEN_BANDS
        self.ground_cover = ground_cover
        self.data_percent = data_percent
        self.use_satmae = use_satmae

        if bands:
            assert all(band in BEN_BANDS for band in bands)
            exclude_bands =  list(set(BEN_BANDS) - set(bands))
            self.exclude_re = re.compile(f".*({'|'.join(exclude_bands)})\.tif$")
            #self.stat_idxs = [ind for ind, val in enumerate(BEN_BANDS)
            #                    if val in bands] 

        if self.splits_to_use == "torchgeo":
            self.mean = np.array([BEN_TG_TRAIN_MEAN[ind]
                                        for ind in self.stat_idxs]).reshape(1, 1, -1)
            self.std = np.array([BEN_TG_TRAIN_STD[ind]
                                        for ind in self.stat_idxs]).reshape(1, 1, -1)
            self.scaled_mean = np.array([BEN_TG_SCALED_TRAIN_MEAN[ind]
                                        for ind in self.stat_idxs])
            self.scaled_std = np.array([BEN_TG_SCALED_TRAIN_STD[ind]
                                        for ind in self.stat_idxs])
        elif self.splits_to_use == "tensorflow":
            self.mean = [BEN_MEAN[ind]
                                        for ind in self.stat_idxs]
            self.std = [BEN_STD[ind]
                                        for ind in self.stat_idxs]
            self.scaled_mean = [BEN_SCALED_MEAN[ind]
                                        for ind in self.stat_idxs]
            self.scaled_std = [BEN_SCALED_STD[ind]
                                        for ind in self.stat_idxs]
        else:
            raise Exception("Only 'torchgeo' and 'tensorflow' are valid split options")

        # For USatMAE training, we exclude all 60m band data, so we explicitly
        # remove those TODO: don't hardcode and dynamically change it
        self.fixed_img_transforms = self._get_transform(satmae_transform, standardize)

        # BigEarthNet dataset source seems to have no (or an expired) 
        # SSL certificate causing download issues. This line can be removed
        # once it is fixed (also ssl import can be removed)
        ssl._create_default_https_context = ssl._create_unverified_context

        self._verify()
        self.bad_patches = self._load_bad_patches(exclude_bad_patches)
        self.folders = self._load_folders()
 
    def _get_transform(self, satmae_transform: bool, standardize: bool) -> Callable:
        """
        Creates a callable comprised of image transforms specified in SatMAE
        paper
        """
        custom_transforms = {}

        reduced_map = [band for band in self.band_map if band in self.ben_bands]

        if self.use_satmae:
            img_sizes = [self.image_size for band in reduced_map]
        else:
            img_sizes = [self.ground_cover / self.gsd_map[band] for band in reduced_map]
        cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(self.stat_idxs))
        cons_vert_flip = ConsistentRadomVerticalFlip(len(self.stat_idxs))

        for idx, band in enumerate(reduced_map):
            img_transforms = []
            img_transforms.append(SatMAENormalize(self.mean[idx], self.std[idx]))
            img_transforms.append(tv_transform.ToTensor())

            if satmae_transform:
                img_transforms.extend(get_satmae_transforms(self.split, self.image_size))

            if standardize:
                img_transforms.append(tv_transform.Normalize(self.scaled_mean[idx], self.scaled_std[idx]))

            if self.split == "train":
                # Apply consistent horizontal and vertical rotation
                img_transforms.append(cons_rand_crop)
                img_transforms.append(cons_horiz_flip)
                img_transforms.append(cons_vert_flip)
            else:
                img_transforms.append(tv_transform.CenterCrop(img_sizes[idx]))
            
            custom_transforms[band] = tv_transform.Compose(img_transforms)

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
        """Override _load_image to apply custom transform at load
        stage before converted to a Torch tensor.
        
        Keyword arguments:
            index: index to return
        Returns:
            tensor: the raster image or target
        """
        paths, band_order = self._load_paths(index)

        # Save file paths based on saved data
        imgs = {}
        # Need to apply fixed_img_transforms in the same order as band_map
        reduced_map = [band for band in self.band_map if band in self.ben_bands]
        for band_name in reduced_map:
            idx = band_order.index(band_name)
            path = paths[idx]
            if band_name not in ['B01', 'B09']:
                # Don't resample until after loading is done
                with rasterio.open(path) as dataset:
                    array = dataset.read(
                        indexes=1,
                        out_dtype="int32",
                    )
                    # Rescale based on GSD for the band. 10m comes out as 120x120
                    # 20m comes out as 60x60, 60m comes out as 20x20
                    imgs[self.band_map[band_name]] = self.fixed_img_transforms[band_name](array)
                    #images.append(array)
        #arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        #arrays = arrays.transpose(1, 2, 0).astype(np.float32)
        #tensor = self.fixed_img_transforms(arrays)

        #imgs = {}
        #for idx in self.stat_idxs:
        #    band_name = band_order[idx]
        #    if band_name not in ['B01', 'B09']:
        #        imgs[self.band_map[band_name]] = tensor[idx, :, :]

        return imgs

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Override __getitem__ to apply transform only to images and not
        labels.

        Keyword arguemnts:
            index: index to return
        Returns:
            (image, label): image and corresponding label for given index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        return image, label

    def _load_paths(self, index: int) -> List[str]:
        """Override _load_paths to drop unused bands by
        omitting them from indexed paths.

        Keyword arguments:
            index: index to return
        Returns:
            paths: list of file paths
        """
        paths = super()._load_paths(index)
        # Drop bands that are not needed
        filtered_paths = []
        all_band_order = []
        for path in paths:
            # Add the band regardless of if its filterd out b/c the 
            # indexing done in init already does that
            all_band_order.append(self.band_regex.match(path).group(1))
            if self.exclude_re and self.exclude_re.match(path):
                continue
            filtered_paths.append(path)
        return filtered_paths, all_band_order

    def _load_folders(self) -> List[Dict[str, str]]:
        """Override _load_folders to perform dataset
        reduction during indexing rather than runtime.

        Returns:
            folders: List[Dict[str, str]] - list of folders
        """

        # TorchGeo dataset has s1 and s2 data in the 
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        # If the CSV file being used is from TensorFlow, it 
        # won't have both S1 and S2 data, assume its only S2 data
        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        if self.splits_to_use == "tensorflow":
            folders = [
                {
                    "s1": os.path.join(self.root, dir_s1, pair[1]),
                    "s2": os.path.join(self.root, dir_s2, pair[0])
                }
                for pair in pairs if pair[0] not in self.bad_patches
            ]
        # TorchGeo seems to already filter out all the bad patches
        elif self.splits_to_use == "torchgeo":
            folders = [
                {
                    "s1": os.path.join(self.root, dir_s1, pair[1]),
                    "s2": os.path.join(self.root, dir_s2, pair[0]),
                }
                for pair in pairs if pair
            ]

        if self.split in ['train', 'val']:
            # SatMAE uses 10% of train dataset
            # Random sampling function written to match SeCo
            # TODO: find a way to dynamically get seed into dataset
            # b/c default_rng call doesn't follow from global seeding
            rng = np.random.default_rng(42)
            indices = rng.choice(range(len(folders)), int(self.data_percent * len(folders)))
            folders = np.array(folders)[indices].tolist()
        return folders

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
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
        #collated_imgs = []
        band_map_vals = [band_name for band, band_name in self.band_map.items() if band in self.ben_bands]
        for band in band_map_vals:
            collated_imgs[band] = torch.stack([img[band] for img in imgs])
        #collated_imgs = torch.stack(collated_imgs, dim=1)
        labels = torch.stack(labels)
        return collated_imgs, labels


if __name__ == '__main__':
    train_ben_dataset = BigEarthNetDatasetUSat(split='train', base_path='/scr/bigearthnet', 
                                               sentinel_source='all', satmae_transform=False,
                                               standardize=True, download=True, ground_cover=640, data_percent=0.1)
    print(len(train_ben_dataset))
    print(train_ben_dataset[0][0]['S2:Red'].shape)
    print(train_ben_dataset[0][0]['S2:SWIR1'].shape)

    # Can be uncommented to generate dataset stats
    #from usat.utils.helper import compute_dataset_stats
    #compute_dataset_stats(train_ben_dataset, num_channels=2)

def map_s1_s2_files(directory):
    from pathlib import Path
    import json
    files = Path(directory).glob('*/*labels_metadata.json')
    s2_to_s1_map = {}
    for file in files:
        s1_dir_name = str(file.absolute()).split('/')[4]
        with open(file, 'r') as f:
            metadata = json.load(f)
            s2_to_s1_map[metadata["corresponding_s2_patch"]] = s1_dir_name

    with open('/scr/bigearthnet/bigearthnet-tf-test-mod.csv', 'w') as f1:
        with open('/scr/bigearthnet/bigearthnet-tf-test.csv', 'r') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.strip()
                f1.write(f"{line},{s2_to_s1_map[line]}\n")
    with open('/scr/bigearthnet/bigearthnet-tf-train-mod.csv', 'w') as f1:
        with open('/scr/bigearthnet/bigearthnet-tf-train.csv', 'r') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.strip()
                f1.write(f"{line},{s2_to_s1_map[line]}\n")
    with open('/scr/bigearthnet/bigearthnet-tf-val-mod.csv', 'w') as f1:
        with open('/scr/bigearthnet/bigearthnet-tf-val.csv', 'r') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.strip()
                f1.write(f"{line},{s2_to_s1_map[line]}\n")


#map_s1_s2_files('/scr/bigearthnet/BigEarthNet-S1-v1.0')


