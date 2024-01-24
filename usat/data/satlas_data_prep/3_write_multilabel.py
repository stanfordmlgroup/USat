import json
import pandas as pd
import os
import sys
from PIL import Image
import torch
from torchvision import transforms
from satlas_utils import geo_to_mercator, mercator_to_geo
from shapely.geometry import Polygon, LineString, Point
import multiprocessing

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

segmentation_categories = ['invalid', 'water', 'developed', 'tree', 'shrub', 'grass', 'crop_cover', 'bare', 'snow', 'wetland', 'mangroves', 'moss']


def get_labels(df,idx,satlas_data_path,usat_data_path,split,naip_grid_label_file):

    s2_uuid = df.loc[idx]['UUID_s2']
    s2_grid = df.loc[idx]['s2_grid']
    naip_grid = df.loc[idx]['naip_grid']
    naip_uuid = df.loc[idx]['UUID_naip']
    
    # get labels from land cover segmentation mask
    seg_path = os.path.join(usat_data_path, split, 'land_cover_cropped_resized', naip_grid + '.png')
    image = Image.open(seg_path)
    transform = transforms.ToTensor()
    seg_tensor = transform(image) 
    seg_tensor = seg_tensor*255 # value between 0 to 11
    multilabel_set = set([LABEL_MAP[segmentation_categories[int(x)]] for x in seg_tensor.unique().tolist()])
    
    # get labels from object detection file vector.json
    base_label_path = os.path.join(satlas_data_path,f'{split}_labels',s2_grid,'label_0','vector.json')
    
    if not os.path.exists(base_label_path): 
        # case when object detection label not available, only write segmentation labels
        output = naip_grid+','+'_'.join([str(x) for x in multilabel_set])+'\n'
        with open(naip_grid_label_file, 'a') as f:
            f.write(output)
        return
    
    f = open(base_label_path)
    poly_label = json.load(f)
    # relative position of the NAIP grid in the S2 grid
    naip_col, naip_row = int(naip_grid.split('_')[0]), int(naip_grid.split('_')[1])
    s2_col, s2_row = int(s2_grid.split('_')[0]), int(s2_grid.split('_')[1])
    s2_ul_naipgrid = geo_to_mercator(mercator_to_geo((s2_col,s2_row),pixels=1,zoom=13), pixels=1,zoom=17)
    # upper left corner of the NAIP tile, in the S2 tile's coord system
    ul_coord = [round((naip_col-s2_ul_naipgrid[0])*512),round((naip_row-round(s2_ul_naipgrid[1]))*512)]
    naip_crop_polygon = Polygon([ul_coord, [ul_coord[0]+512,ul_coord[1]], [ul_coord[0]+512,ul_coord[1]+512], [ul_coord[0],ul_coord[1]+512]])
    
    for label_key in poly_label.keys():
        if label_key in LABEL_MAP:
            for obj in poly_label[label_key]:
                polytype = obj['Geometry']['Type'].capitalize()
                poly_coord = obj['Geometry'][polytype]
                if polytype == 'Polygon':
                    poly_coord = poly_coord[0]
                    obj_polygon = Polygon(poly_coord)
                elif polytype == 'Polyline':
                    obj_polygon = LineString(poly_coord)
                elif polytype == 'Point':
                    obj_polygon = Point(poly_coord)
                if obj_polygon.intersects(naip_crop_polygon):
                    multilabel_set.add(LABEL_MAP[label_key])
                    break
    output = naip_grid+','+'_'.join([str(x) for x in multilabel_set])+'\n'
    with open(naip_grid_label_file, 'a') as f:
        f.write(output)
    return


    
def process_dataframe_chunk(df_chunk, output_queue, satlas_data_path, usat_data_path, split, naip_grid_label_file):
    for idx in df_chunk.index:
        get_labels(df_chunk, idx, satlas_data_path, usat_data_path, split, naip_grid_label_file)



if __name__ == '__main__':

    satlas_data_path = sys.argv[1]
    usat_data_path = sys.argv[2]
    num_workers = int(sys.argv[3])

    for split in ['train','val']:
        print(f"start writing {split} metadata")

        if split == 'train':
            df = pd.read_csv(os.path.join(usat_data_path,'paired_metadata_with_landcover.csv'))
            naip_grid_label_file = os.path.join(usat_data_path,'naip_grid_multilabel.csv')
            outfile = os.path.join(usat_data_path,'paired_metadata_multilabel.csv')
        else:
            df = pd.read_csv(os.path.join(usat_data_path,'paired_metadata_val_with_landcover.csv'))
            naip_grid_label_file = os.path.join(usat_data_path,'naip_grid_multilabel_val.csv')
            outfile = os.path.join(usat_data_path,'paired_metadata_val_multilabel.csv')
        
        with open(naip_grid_label_file, 'w') as f:
            f.writelines('naip_grid,labels\n')
        

        # Step 1: obtain the labels for each NAIP tile
        chunk_size = len(df) // num_workers
        output_queue = multiprocessing.Queue()
        processes = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(df)
            df_chunk = df.iloc[start_idx:end_idx]
            process = multiprocessing.Process(target=process_dataframe_chunk, args=(df_chunk, output_queue, satlas_data_path, usat_data_path, split, naip_grid_label_file))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        print(f"finished writing {split} NAIP multilabels")

        # Step 2: append multilabel column to metadata
        multi_lab = pd.read_csv(naip_grid_label_file)
        multi_lab_dedup = multi_lab.drop_duplicates()
        merged_df = df.merge(multi_lab_dedup, on = "naip_grid", how = 'inner', suffixes = ['','_1'])
        merged_df.to_csv(outfile,index = False)
        print(f"finished writing {split} metadata")
        


