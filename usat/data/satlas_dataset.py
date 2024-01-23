import os
import random
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from usat.utils.builder import DATASET
from usat.utils.sentinel import *
 

@DATASET.register_module()
class SatlasDataset(Dataset):

    # statistics used for normalization, calculated using 10% samples
    MEAN = {'NAIP:Red': 0.4004, 'NAIP:Green': 0.4406, 'NAIP:Blue': 0.3875, 'S2:Red': 0.3433, 'S2:Green': 0.3639, 'S2:Blue': 0.3845, 'S2:RE1': 0.8791, 'S2:RE2': 0.9180, 'S2:RE3': 0.9203, 'S2:NIR': 0.9138, 'S2:SWIR1': 0.8912, 'S2:SWIR2': 0.8039}
    STD = {'NAIP:Red': 0.2208, 'NAIP:Green': 0.2039, 'NAIP:Blue': 0.1897, 'S2:Red': 0.2417, 'S2:Green': 0.2086, 'S2:Blue': 0.2016, 'S2:RE1': 0.2471, 'S2:RE2': 0.2432, 'S2:RE3': 0.2429, 'S2:NIR': 0.2517, 'S2:SWIR1': 0.2816, 'S2:SWIR2': 0.3097}
    GSDS = {'NAIP:Red': 1.0, 'NAIP:Green': 1.0, 'NAIP:Blue': 1.0, 'S2:Red': 10.0, 'S2:Green': 10.0, 'S2:Blue': 10.0, 'S2:RE1': 20.0, 'S2:RE2': 20.0, 'S2:RE3': 20.0, 'S2:NIR': 10.0, 'S2:SWIR1': 20.0, 'S2:SWIR2': 20.0, 'S2:LABEL': 10.0}

    s2_bandname_map = {'RE1':'b05', 'RE2':'b06', 'RE3':'b07', 'NIR':'b08', 'SWIR1':'b11', 'SWIR2':'b12', 'Red':'tci', 'Green':'tci', 'Blue':'tci'}
    rgb_channel_id = {'Red':0, 'Green':1, 'Blue':2}

    LABEL_MAP = {'aerialway_pylon': 0,
    'airport': 1,
    'airport_apron': 2,
    'airport_hangar': 3,
    'airport_runway': 4,
    'airport_taxiway': 5,
    'airport_terminal': 6,
    'aquafarm': 7,
    'bare': 8,
    'chimney': 9,
    'communications_tower': 10,
    'crop': 11,
    'crop_cover': 12,
    'dam': 13,
    'developed': 14,
    'flagpole': 15,
    'fountain': 16,
    'gas_station': 17,
    'grass': 18,
    'helipad': 19,
    'invalid': 20,
    'landfill': 21,
    'lighthouse': 22,
    'lock': 23,
    'mangroves': 24,
    'mineshaft': 25,
    'moss': 26,
    'offshore_platform': 27,
    'offshore_wind_turbine': 28,
    'park': 29,
    'parking_garage': 30,
    'parking_lot': 31,
    'petroleum_well': 32,
    'pier': 33,
    'power_plant': 34,
    'power_substation': 35,
    'power_tower': 36,
    'quarry': 37,
    'raceway': 38,
    'railway': 39,
    'river': 40,
    'road': 41,
    'shrub': 42,
    'silo': 43,
    'ski_resort': 44,
    'snow': 45,
    'solar_farm': 46,
    'stadium': 47,
    'storage_tank': 48,
    'theme_park': 49,
    'toll_booth': 50,
    'track': 51,
    'tree': 52,
    'wastewater_plant': 53,
    'water': 54,
    'water_tower': 55,
    'wetland': 56,
    'wind_turbine': 57}

    VALID_LABELS = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 29, 30, 
    31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

    def __init__(self,
                 base_path: str = None,
                 split: str = 'train',
                 standardize: bool = True,
                 full_return: bool = False,
                 custom_transform = None,
                 pretrain: bool = False,
                 ground_cover: int = 240,
                 data_percent: float = 1.0,
                 pad: bool = False,
                 discard_bands: list = [],
                 target_type = 'multilabel', # 'multilabel' or 'segmentation'
                 use_valid_labels = False, # whether to use 44 classes instead of 58
                 target_gsd = 10 # change this to resize segmentation mask resolution
                 ):

        if base_path is None:
            base_path = '/scr/satlas_full'
        self.base_path = base_path
        self.split = split
        self.standardize = standardize
        self.full_return = full_return
        self.pretrain = pretrain
        self.ground_cover = ground_cover
        self.pad = pad
        self.discard_bands = discard_bands
        self.target_type = target_type
        self.target_gsd = target_gsd
        self.use_valid_labels = use_valid_labels

        # Sort bands from largest to smallest GSD
        self.all_bands = [band[0] for band in sorted(self.GSDS.items(), key=lambda x: x[1], reverse=True)] # including label

        self.custom_transform = self._build_transforms()

        if self.split == "train":
            self.metadata = pd.read_csv(os.path.join(self.base_path,'paired_metadata_multilabel.csv')).sample(frac=data_percent,random_state = 42)
        else:
            self.metadata = pd.read_csv(os.path.join(self.base_path,'paired_metadata_val_multilabel.csv')).sample(frac=data_percent,random_state = 42)
        

    def _build_transforms(self):
        custom_transforms = {} # keys: "product:band", values: torch transform
        img_size_smallest = int(self.ground_cover / self.GSDS[self.all_bands[0]])
        img_sizes = [int(img_size_smallest*self.GSDS[self.all_bands[0]]/self.GSDS[band]) for band in self.all_bands]
        cons_rand_crop = ConsistentRandomCrop(img_sizes, pad_if_needed=True, padding_mode='constant', fill=0)
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(img_sizes))
        cons_vert_flip = ConsistentRadomVerticalFlip(len(img_sizes))

        for idx, product_band in enumerate(self.all_bands):
            product, band = product_band.split(':')
            t = []

            if self.standardize and band != 'LABEL':
                t.append(transforms.Normalize(self.MEAN[product_band], self.STD[product_band]))
            
            if self.split == 'train':
                t.append(cons_rand_crop)
                t.append(cons_horiz_flip)
                t.append(cons_vert_flip)
            else:
                t.append(transforms.CenterCrop(img_sizes[idx]))

            custom_transforms[product_band] = transforms.Compose(t)
        return custom_transforms

    def __len__(self):
        return len(self.metadata)    

    def _load_img(self, idx, product_band):
        product, band = product_band.split(':')
        s2_uuid = self.metadata.iloc[idx]['UUID_s2']
        s2_grid = self.metadata.iloc[idx]['s2_grid']
        naip_grid = self.metadata.iloc[idx]['naip_grid']
        naip_uuid = self.metadata.iloc[idx]['UUID_naip']
        if product == 'NAIP':
            img_path = os.path.join(self.base_path, self.split, 'highres_resized', naip_uuid, 'tci', naip_grid+'.png')
        else:
            if band == 'LABEL':
                img_path = os.path.join(self.base_path, self.split, 'land_cover_cropped_resized', naip_grid+'.png')
            else:
                img_path = os.path.join(self.base_path, self.split, 's2_cropped_resized', s2_uuid, self.s2_bandname_map[band], naip_grid+'.png')
        image = Image.open(img_path)

        transform = transforms.ToTensor()
        tensor = transform(image) # (3,H,W), 0-1 value
        channel_idx = self.rgb_channel_id.get(band, 0)
        tensor = tensor[channel_idx][None,:,:] # (1,H,W)
        if product_band == 'S2:LABEL':
            tensor = tensor*255

        return self.custom_transform[product_band](tensor)

    # def _load_label(self, idx):
    #     s2_uuid = self.metadata.iloc[idx]['UUID_s2']
    #     naip_grid = self.metadata.iloc[idx]['naip_grid']
    #     img_path = os.path.join(self.base_path, self.split, 'land_cover_cropped', naip_grid+'.png')
    #     image = Image.open(img_path)
    #     transform = transforms.ToTensor()
    #     tensor = (transform(image)*255)[0][None,:,:]

    #     return self.custom_transform['S2:LABEL'](tensor)

    
    def __getitem__(self, idx):
        items = {}

        for product_band in self.all_bands:
            product, band = product_band.split(':')
            img = self._load_img(idx, product_band)
            if product_band not in self.discard_bands:
                items[product_band] = img
        
        label = items.pop('S2:LABEL')
        if self.target_gsd != 10:
            target_size = int(label.shape[-1]*10//self.target_gsd)
            target_transform = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)
            label = target_transform(label)
        
        if self.target_type == 'multilabel':
            class_idxs = [int(x.strip()) for x in self.metadata.iloc[idx]['labels'].split('_')]
            label = torch.zeros(len(self.LABEL_MAP), dtype=torch.int)
            label[class_idxs] = 1
            if self.use_valid_labels:
                label = label[self.VALID_LABELS]

        if self.full_return:
            return items, label, idx
        return items, label


    def collate_fn (self, batch):
        if self.full_return:
            imgs, labels, paths = zip(*batch)
        else:
            imgs, labels = zip(*batch)
        imgs_batch = {}
        for product_band in self.all_bands:
            if product_band not in self.discard_bands and product_band != 'S2:LABEL':
                imgs_batch[product_band] = torch.stack([img[product_band] for img in imgs]) # bs,1,H,W
        labels =torch.stack(labels) # bs,1,H,W
        if self.full_return:
            return imgs_batch, labels, paths
        else:
            return imgs_batch, labels


if __name__ == '__main__':
    satlasdata = SatlasDataset(split="train", standardize = False, ground_cover = 240, pad = True, discard_bands = ['NAIP:Red', 'NAIP:Green', 'NAIP:Blue'])
    sample = satlasdata[1000]
    img_dict = sample[0]
    label = sample[1]
    for key in img_dict.keys():
        print(key, img_dict[key].shape)
    print('label', label.shape)

    from torch.utils.data import DataLoader
    dl = DataLoader(satlasdata, batch_size=4, num_workers=0)
    for idx, item in enumerate(dl):
        x,y = item 
        for key in x.keys():
            print(key, x[key].shape)
        print('label', y.shape)
        break
    
    # visualize S2 rgb, NAIP rgb, segmentation mask
    # from skimage import io
    # from skimage import color
    # from skimage import segmentation
    # import matplotlib.pyplot as plt
    # s2_rgb = torch.stack([img_dict['S2:Red'][0],img_dict['S2:Green'][0],img_dict['S2:Blue'][0]])
    # plt.imshow(s2_rgb.permute(1, 2, 0))
    # naip_rgb = torch.stack([img_dict['NAIP:Red'][0],img_dict['NAIP:Green'][0],img_dict['NAIP:Blue'][0]])
    # plt.imshow(naip_rgb.permute(1, 2, 0))
    # io.imshow(color.label2rgb(label.numpy().astype('int')[0]))