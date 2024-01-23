from typing import Callable, Collection
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchgeo.datasets  import EuroSAT
import torchvision.transforms as T 
import numpy as np

from usat.utils.builder import DATASET 
from usat.utils.constants import SENTINEL_2_BANDS
from usat.utils.sentinel import (get_satmae_transforms, SatMAENormalize, ConsistentRadomHorizontalFlip,
                                     ConsistentRadomVerticalFlip, ConsistentRandomCrop)
from usat.utils.constants import EUROSAT_CHANNEL_MEANS, EUROSAT_CHANNEL_STD_DEVS
from usat.utils.constants import EUROSAT_MEANS, EUROSAT_STDS, EUROSAT_SCALED_MEANS, EUROSAT_SCALED_STSDS

""" 
Definition for EuroSat-10 Dataset 
"""
@DATASET.register_module()
class EuroSatDataset(Dataset):
    """
    A thin wrapper around torchgeo's EuroSat dataset
    """
    def __init__(self, 
        base_path: str,
        split: str =  'train',
        standardize: bool = False,
        bands: Collection[str]   = ['B04', 'B03', 'B02', 'B08'],
        custom_transform: Callable = None,
        image_size: int = 224, 
        download: bool = True, 
        satmae_transform: bool = True) -> None:
        """
        base_path: Path where EuroSAT is downloaded to by torchgeo
        split: Which data to use ('train/val/test')
        standardize: Apply Torch.Normalize on top of existing transformations
        bands: The order of bands to use. Default is RGB NIR 
        custom_transform: Any extra transforms to apply to the data at the end of the pipeline
        image_size: Size to scale images too 
        download: Whether to download the dataset if it doesn't exist at the path
        satmae_transform: Use the same image transforms as SatMae (ignore standardize)
        """
        self.bands = [SENTINEL_2_BANDS.index(band) for band in bands]

        band_transform = T.Lambda(lambda image: image[self.bands, : , :])
        transforms = [band_transform]

        means = [EUROSAT_CHANNEL_MEANS[i] for i in self.bands]
        means = np.array(means)
        means = means.reshape((-1, 1, 1))
        std_devs = np.array([EUROSAT_CHANNEL_STD_DEVS[i] for i in self.bands]).reshape((-1, 1, 1 ))

        transforms.append(SatMAENormalize(means, std_devs, 1.0))
        if satmae_transform:
            transforms.extend(get_satmae_transforms(split, image_size))

        if standardize:
            transforms.append(T.Normalize(means, std_devs))
        
        if custom_transform is not None:
            transforms.append(custom_transform)
       
        self.image_transform = T.Compose(transforms)

        self.dataset = EuroSAT(base_path,  split, 
            transforms = torchvision.transforms.Lambda(self.__transform_wrapper__),
            download = download)
        

    def __len__(self): 
        return len(self.dataset) 

    def __getitem__(self, index): 
        return self.dataset[index] 

    def __transform_wrapper__(self, object):   
        """
        Wrapper transform to pack and unpack from dictionary
        """
        image, label = object['image'], object['label'] 
        
        assert image.shape[0] < image.shape[1] and image.shape[1] == image.shape[2] 
        image = self.image_transform(image).float()

        return (image, label) 


import ssl

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
    """
    A thin wrapper around torchgeo's EuroSat dataset
    """ 
    band_map = {
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

    gsd_map = {
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
        bands: Collection[str] = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12",],
        custom_transform: Callable = None,
        ground_cover: int = 640,
        image_size: int = 224,
        use_satmae: bool = False,
        data_percent: float = 1.0,
        download: bool = False) -> None:
        """
        base_path: Path where EuroSAT is downloaded to by torchgeo
        split: Which data to use ('train/val/test')
        standardize: Apply Torch.Normalize on top of existing transformations
        bands: The order of bands to use. Default is RGB NIR 
        custom_transform: Any extra transforms to apply to the data at the end of the pipeline
        image_size: Size to scale images too 
        use_satmae: Whether to resize images 
        download: Whether to download the dataset if it doesn't exist at the path
        """
        super().__init__(
            root=base_path,
            split=split,
            bands=bands,
            transforms=torchvision.transforms.Lambda(self.__transform_wrapper__),
            download=download
        )

        # Re-run parent class constructors to allow for subsets of data
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

        self.band_names = [self.band_map[b] for b in bands]
        self.band_gsds = [self.gsd_map[b] for b in bands]
        self.ground_cover = ground_cover



        self.img_transforms = {}
        # Don't use 60m bands
        reordered_band_idxs = [self.all_band_names.index(b) for b in self.band_map if b not in ['B01', 'B09', 'B10']]
        if use_satmae:
            img_sizes = [image_size for idx in reordered_band_idxs]
        else:
            img_sizes = [self.ground_cover // self.gsd_map[self.all_band_names[idx]] for idx in reordered_band_idxs]
        cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(img_sizes))
        cons_vert_flip = ConsistentRadomVerticalFlip(len(img_sizes))
        for order, idx in enumerate(reordered_band_idxs):
            img_transform = []
            img_transform.append(SatMAENormalize(EUROSAT_MEANS[idx], EUROSAT_STDS[idx], 1.0))

            if standardize:
                img_transform.append(T.Normalize(EUROSAT_SCALED_MEANS[idx], EUROSAT_SCALED_STSDS[idx]))

            # Resize, specifically for 20m and 60m
            img_transform.append(T.Resize(img_sizes[order], interpolation=T.InterpolationMode.BICUBIC))

            if split == "train":
                img_transform.append(cons_rand_crop)
                img_transform.append(cons_horiz_flip)
                img_transform.append(cons_vert_flip)
            else:
                img_transform.append(T.CenterCrop(img_sizes[order]))
        
            self.img_transforms[self.band_map[self.all_band_names[idx]]] = T.Compose(img_transform)

    def __transform_wrapper__(self, object):   
        """
        Wrapper transform to pack and unpack from dictionary
        """
        image, label = object['image'], object['label'] 

        reordered_band_idxs = [self.all_band_names.index(b) for b in self.band_map if b not in ['B01', 'B09', 'B10']]
        imgs = {}
        for idx in reordered_band_idxs:
            band_name = self.band_map[self.all_band_names[idx]]
            if self.all_band_names[idx] not in ["B01", "B09", "B10"]:
                imgs[band_name] = self.img_transforms[band_name](image[idx, :, :][None, :, :])

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

    print(len(final_files))

    with open(os.path.join(root, f"eurosat-{split}_{data_percent}.txt"), 'w') as f:
        f.writelines(final_files)


if __name__ == '__main__':
    train_eurosat_dataset = EuroSatDatasetUSat(base_path= '/scr/eurosat', standardize=True, use_satmae=True, image_size=96, download=True, data_percent=1.0)
    print(len(train_eurosat_dataset))

    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.0062)
    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.06173)
    #create_partial_dataset('/scr/eurosat', 'train', data_percent=0.617284)
