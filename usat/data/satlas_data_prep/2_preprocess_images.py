"""
This script preprocess all images and segmentation labels and saves to file using a dataloader instance
"""

import os
import sys
import cv2
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from satlas_utils import geo_to_mercator, mercator_to_geo

 

class DummySatlasDataset(Dataset):

    GSDS = {'NAIP:Red': 0.597, 'NAIP:Green': 0.597, 'NAIP:Blue': 0.597, 'S2:Red': 9.552, 'S2:Green': 9.552, 'S2:Blue': 9.552, 'S2:RE1': 9.552, 'S2:RE2': 9.552, 'S2:RE3': 9.552, 'S2:NIR': 9.552, 'S2:SWIR1': 9.552, 'S2:SWIR2': 9.552, 'S2:LABEL': 9.552}

    s2_bandname_map = {'RE1':'b05', 'RE2':'b06', 'RE3':'b07', 'NIR':'b08', 'SWIR1':'b11', 'SWIR2':'b12', 'Red':'tci', 'Green':'tci', 'Blue':'tci'}
    rgb_channel_id = {'Red':0, 'Green':1, 'Blue':2}

    def __init__(self,
                 base_path: str = None,
                 split: str = 'train',
                 out_path: str = None
                 ):

        if base_path is None:
            base_path = '/deep2/group/aicc-bootcamp/self-sup/satlas'
        self.base_path = base_path
        self.split = split
        self.out_path = out_path

        # List of "product:band" including segmentation label, in descending GSDS order
        self.all_bands = [band[0] for band in sorted(self.GSDS.items(), key=lambda x: x[1], reverse=True)]

        # Create paths to the processed images
        self.naip_new_dir = os.path.join(self.out_path,self.split, 'highres_resized')
        self.s2_new_dir = os.path.join(self.out_path,self.split, 's2_cropped_resized')
        self.label_new_dir = os.path.join(self.out_path,self.split, 'land_cover_cropped_resized')
        for path in [self.naip_new_dir,self.s2_new_dir,self.label_new_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Load metadata
        if self.split == "train":
            self.metadata = pd.read_csv(os.path.join(usat_data_path,'paired_metadata_with_landcover.csv'))
        else:
            self.metadata = pd.read_csv(os.path.join(usat_data_path,'paired_metadata_val_with_landcover.csv'))


    def __len__(self):
        return len(self.metadata)    


    def _process_naip(self, naip_uuid, naip_grid):
        naip_path = os.path.join(self.base_path, self.split, 'highres', naip_uuid, 'tci', naip_grid+'.png')
        naip_uuid_new_dir = os.path.join(self.naip_new_dir, naip_uuid, 'tci')
        if not os.path.exists(naip_uuid_new_dir):
            try: 
                os.makedirs(naip_uuid_new_dir)
            except:
                pass # to avoid path exists error caused by multi-processing
        naip_new_path = os.path.join(naip_uuid_new_dir,naip_grid+'.png')
        if not os.path.exists(naip_new_path):
            naipimage = cv2.imread(naip_path)
            resized = cv2.resize(naipimage, (300,300), interpolation = cv2.INTER_LINEAR) # bilinear, 1m gsd
            cv2.imwrite(naip_new_path, resized)


    def _process_s2(self, band, s2_uuid, s2_grid, naip_uuid, naip_grid):
        # define the source destination of the S2 image
        if band != 'LABEL':
            s2_path = os.path.join(self.base_path, self.split, 'images', s2_uuid, self.s2_bandname_map[band], s2_grid+'.png')
            s2_uuid_band_new_dir = os.path.join(self.s2_new_dir, s2_uuid, self.s2_bandname_map[band])
            if not os.path.exists(s2_uuid_band_new_dir):
                try:
                    os.makedirs(s2_uuid_band_new_dir)
                except:
                    pass 
            s2_new_path = os.path.join(s2_uuid_band_new_dir,naip_grid+'.png')
        else:
            s2_path = os.path.join(self.base_path, f'{split}_labels', s2_grid,'label_0','land_cover.png')
            s2_new_path = os.path.join(self.label_new_dir,naip_grid+'.png')

        if not os.path.exists(s2_new_path):
            # define the cropping area
            naip_col, naip_row = int(naip_grid.split('_')[0]), int(naip_grid.split('_')[1])
            s2_col, s2_row = int(s2_grid.split('_')[0]), int(s2_grid.split('_')[1])
            # the upper left corner of the S2 image in zoom level 17
            s2_ul_naipgrid = geo_to_mercator(mercator_to_geo((s2_col,s2_row),pixels=1,zoom=13), pixels=1,zoom=17)
            crop_pixel_col = round((naip_col-s2_ul_naipgrid[0])/16*512)
            crop_pixel_row = round((naip_row-round(s2_ul_naipgrid[1]))/16*512)
            # crop the raw image
            s2image = cv2.imread(s2_path)
            s2image_cropped = s2image[crop_pixel_row:crop_pixel_row+32,crop_pixel_col:crop_pixel_col+32]
            # resize to 10m and 20m gsd
            if band == 'LABEL':
                resized = cv2.resize(s2image_cropped, (30,30), interpolation = cv2.INTER_NEAREST) # nearest, 10m gsd
            elif self.s2_bandname_map[band] in ['b05',  'b06',  'b07',  'b11',  'b12']:
                resized = cv2.resize(s2image_cropped, (15,15), interpolation = cv2.INTER_LINEAR) # bilinear, 20m gsd
            else:
                resized = cv2.resize(s2image_cropped, (30,30), interpolation = cv2.INTER_LINEAR) # bilinear, 10m gsd
            cv2.imwrite(s2_new_path, resized)


    def _load_img(self, idx, product_band):
        product, band = product_band.split(':')
        s2_uuid = self.metadata.iloc[idx]['UUID_s2']
        s2_grid = self.metadata.iloc[idx]['s2_grid']
        naip_uuid = self.metadata.iloc[idx]['UUID_naip']
        naip_grid = self.metadata.iloc[idx]['naip_grid']

        if product == 'NAIP':
            self._process_naip(naip_uuid, naip_grid)
        else:
            self._process_s2(band,s2_uuid,s2_grid,naip_uuid,naip_grid)

        return torch.zeros(1) 
    
    def __getitem__(self, idx):
        for product_band in self.all_bands:
            self._load_img(idx, product_band)
        return torch.zeros(1), torch.zeros(1) 


if __name__ == '__main__':
    satlas_data_path = sys.argv[1]
    usat_data_path = sys.argv[2]
    num_workers = int(sys.argv[3])

    for split in ['train','val']:
        satlasdata = DummySatlasDataset(base_path = satlas_data_path, split=split, out_path = usat_data_path)
        dl = DataLoader(satlasdata, batch_size=1, num_workers=num_workers)
        for idx, item in enumerate(tqdm(dl)):
            x,y = item 
        print(f'Finished processing {split} split')