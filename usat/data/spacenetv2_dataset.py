import csv
import glob
import os
import random
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import torch
from torch import Tensor
from torchgeo.datasets import SpaceNet2
from torchvision import transforms as tv_transform

from usat.utils.builder import DATASET
from usat.utils.constants import GENERAL_DATA_DIR
from usat.utils.helper import csv_to_dict_list, dict_list_to_csv

SN_TRAIN_MIN = [0., 0., 0.]
SN_TRAIN_MAX = [1781., 1953., 1502.]
SN_TRAIN_MEAN = [331.2314, 463.4669, 357.1541]
SN_TRAIN_STD = [193.3428, 222.7093, 147.1637]
SN_TRAIN_SCALED_MEAN = [0.18598057, 0.23731024, 0.23778569]
SN_TRAIN_SCALED_STD = [0.10855856, 0.11403446, 0.0979785]

@DATASET.register_module()
class SpaceNetV2Dataset(SpaceNet2):
    """Wrapper for the SpaceNetV2 dataset. Train, eval, and test
    splits are based on the original SpaceNet paper. Other data
    filtering and augmentation choices mirror those in the SatMAE
    paper.

    TODO: Add ability to specify api_key from config
    """

    def __init__(self,
                 base_path: str = GENERAL_DATA_DIR,
                 split: str = 'train',
                 standardize: bool = True,
                 custom_transform: Optional[Callable[[Tuple[Tensor, Tensor]],
                                      Tuple[Tensor, Tensor]]] = None,
                 download: bool = True,
                 image: str = "PS-RGB",
                 collections: List[str] = [],
                 api_key: Optional[str] = None,
                 checksum: bool = False) -> None:

        self.spacenetv2_data_dirname = "spacenetv2"
        self.standardize = standardize
        self.img_transforms = [tv_transform.Resize(224)]
        self.split = split
        self.split_r = {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2
        }

        if base_path == GENERAL_DATA_DIR:
            base_path = os.path.join(base_path, self.spacenetv2_data_dirname)

        self.min = torch.Tensor(SN_TRAIN_MIN).reshape(-1, 1, 1)
        self.max = torch.Tensor(SN_TRAIN_MAX).reshape(-1, 1, 1)
        self.mean = torch.Tensor(SN_TRAIN_SCALED_MEAN).reshape(-1, 1, 1)
        self.std = torch.Tensor(SN_TRAIN_SCALED_STD).reshape(-1, 1, 1)

        super().__init__(base_path, image, collections, custom_transform,
                         download, api_key, checksum)

    def _load_files(self, _root: str) -> List[Dict[str, str]]:
        """Override _load_files to allow data loading based on
        data split.

        Keyword arguments:
            root: root dir of dataset
        Returns:
            files: list of dicts with paths for each pair of image and label
        """
        self._handle_split()
        # Load the CSV file corresponding to the current split
        metadata_dir_path = self._curr_metadata_dir_path()
        split_cfg_path = os.path.join(metadata_dir_path, f"{self.split}.csv")
        dict_list = csv_to_dict_list(split_cfg_path)
        for file_path in dict_list:
            # Final path needs to be upated based on image file type used
            file_path.update({"image_path":
                         os.path.join(file_path["image_path"], self.filename)})
        return dict_list

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Override __getitem__ to allow for custom image preprocessing.

        Keyword arugments:
            index: index to return
        Returns:
            (image, label): data and label at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files["image_path"])
        h, w = img.shape[1:]
        mask = self._load_mask(files["label_path"], tfm, raster_crs, (h, w))

        ch, cw = self.chip_size[self.image]
        image = img[:, :ch, :cw]
        mask = mask[None, :ch, :cw]  # Resize requires mask to be 1 x H x W

        # Scale image to be 0-1
        image = image.float()
        image = (image - self.min) / (self.max - self.min)
        image = torch.clip(image, min=0.0, max=1.0)

        for img_transform in self.img_transforms:
            image = img_transform(image)
            mask = img_transform(mask)

        if self.transforms is not None:
            return self.transforms((image, mask))

        # Normalize by mean and std
        if self.standardize:
            image = (image - self.mean) / self.std

        mask = torch.squeeze(mask)

        return (image, mask)

    def _handle_split(self) -> None:
        """SpaceNetV2 TorchGeo implementation doesn't split data
        into train, val, test. This function checks to see if train, val,
        and test configs were generated after data download. If not,
        it generates the config files.
        """
        metadata_dir_path = self._curr_metadata_dir_path()
        metadata_cfgs = [
            os.path.join(metadata_dir_path, "train.csv"),
            os.path.join(metadata_dir_path, "val.csv"),
            os.path.join(metadata_dir_path, "test.csv")
        ]
        if os.path.isdir(metadata_dir_path):
            # Make sure that train, val, and test files exist
            if all([os.path.isfile(path) and os.path.getsize(path) > 0
                    for path in metadata_cfgs]):
                return
        else:
            os.makedirs(metadata_dir_path)

        files = self._get_all_files()
        random.shuffle(files)
        train, val, test = np.split(files, [
            int(len(files) * self.split_r["train"]),
            int(len(files) * (self.split_r["train"] + self.split_r["val"]))])

        # Creates the CSV config files
        dict_list_to_csv(metadata_cfgs[0], train)
        dict_list_to_csv(metadata_cfgs[1], val)
        dict_list_to_csv(metadata_cfgs[2], test)

    def _get_all_files(self) -> List[Dict[str, str]]:
        """Return the path of the image file directories and label files
        in the dataset.

        Returns:
            files: list of dicts with paths for each pair of image and label
        """
        files = []
        for collection in self.collections:
            images = glob.glob(os.path.join(self.root, collection,
                                            "*", self.filename))
            images = sorted(images)
            for imgpath in images:
                imgpath = os.path.dirname(imgpath)
                lbl_path = os.path.join(
                    f"{imgpath}-labels", self.label_glob
                )
                files.append({"image_path": imgpath, "label_path": lbl_path})
        return files

    def _curr_metadata_dir_path(self) -> str:
        """Take the collections being used and determine the unique
        identifier. This allows us to create one set of
        train, val, and test configs per unique grouping rather than
        regenerate them every time. Collection names look like
        "sn2_AOI_2_Vegas", so the locations are being stringed
        together to create the unique ID.

        Returns:
            str: path to metadata dir for configs of this collection subset
        """
        label = sorted([col.split('_')[-1].lower() for col in self.collections])
        return os.path.join(self.root, f"{'_'.join(label)}")


if __name__ == '__main__':
    random.seed(0)
    train_spacenetv2_dataset = SpaceNetV2Dataset(split="train")
    val_spacenetv2_dataset = SpaceNetV2Dataset(split="val")
    test_spacenetv2_dataset = SpaceNetV2Dataset(split="test")
