import os
import pandas as pd
import json
import sys
from satlas_utils import have_all_bands, covered_by_naip, geo_to_mercator, mercator_to_geo

'''
The raw Satlas root path
SATLAS_DATA_PATH
    | train
        | highres
            | UUID
                | tci
                    | grid.png
        | images
            | UUID
                | b05
                | ... 
                    | grid.png
    | val
    | train_labels
    | val_labels
    | metadata


Root path of the processed dataset
USAT_DATA_PATH
    | train
        | highres_resized
            | UUID
                | tci
                    | grid.png
        | s2_cropped_resized
            | UUID
                | b05
                | ...
                    | grid.png
        | land_cover_cropped_resized
            | grid.png
    | val

'''



if __name__ == '__main__':

    satlas_data_path = sys.argv[1]
    usat_data_path = sys.argv[2]

    for split in ['train','val']:
        print(f"start writing {split} metadata")

        # STEP 1: create metadata of all NAIP images
        naip_raw_df = []
        naip_root = os.path.join(satlas_data_path,split,'highres')
        for root, dirs, files in os.walk(naip_root, topdown=False):
            for name in files:
                img_path = os.path.join(root, name)
                relative_path = img_path.removeprefix(satlas_data_path)
                grid = os.path.basename(img_path).removesuffix('.png')
                uid = os.path.basename(os.path.dirname(os.path.dirname(relative_path)))
                naip_raw_df.append({'UUID':uid, 'naip_grid':grid, 'img_path':img_path})
        naip_raw_df = pd.DataFrame.from_records(naip_raw_df)

        # append time column
        f = open(os.path.join(satlas_data_path,"metadata/image_times_naip.json"))
        time_json = json.load(f)
        f.close()
        time_df = pd.DataFrame.from_dict(time_json, orient = 'index')
        time_df.columns = ['timestamp']
        naip_df_joined_dropna = naip_raw_df.join(time_df, on = 'UUID', how = 'inner',rsuffix = 'right')

        print(f"finished writing {split} NAIP metadata")

        # STEP 2: create metadata of all S2 images
        s2_root = os.path.join(satlas_data_path, split, 'images')
        
        # load time metadata
        f = open(os.path.join(satlas_data_path,"metadata/image_times_sentinel2.json"))
        time_json = json.load(f)
        f.close()
        time_df = pd.DataFrame.from_dict(time_json, orient = 'index')
        time_df.columns = ['timestamp']
        timedf_uid_set = set(time_df.index)

        naip_grids_set = set(naip_df_joined_dropna.naip_grid)

        s2_raw_df = []

        for root, dirs, files in os.walk(s2_root, topdown=False):
            for name in files:
                img_path = os.path.join(root, name)
                relative_path = img_path.removeprefix(satlas_data_path)
                uid = os.path.basename(os.path.dirname(os.path.dirname(relative_path)))
                band = os.path.basename(os.path.dirname(relative_path))
                s2_grid = os.path.basename(img_path).removesuffix('.png')
                s2col, s2row = int(s2_grid.split('_')[0]), int(s2_grid.split('_')[1])
                s2_geo_upperleft = mercator_to_geo((s2col, s2row), pixels=1, zoom=13)
                naip_grid_ul = [round(x) for x in geo_to_mercator(s2_geo_upperleft, pixels=1, zoom=17)]
                
                if (band != 'tci') or (not covered_by_naip(naip_grid_ul)) or (uid not in timedf_uid_set) or (not have_all_bands(s2_root,uid,s2_grid)):
                    continue
            
                img_time = time_df.loc[uid]['timestamp']
                
                for col_i in range(naip_grid_ul[0], naip_grid_ul[0]+16):
                    for row_i in range(naip_grid_ul[1], naip_grid_ul[1]+16):
                        if f'{col_i}_{row_i}' in naip_grids_set:
                            s2_raw_df.append({'UUID':uid, 's2_grid':s2_grid, 'naip_grid':f'{col_i}_{row_i}','timestamp':img_time})
        s2_df = pd.DataFrame.from_records(s2_raw_df)

        print(f"finished writing {split} S2 metadata")


        # STEP 3: join S2 and NAIP metadata to make pairs that capture the same ground area
        merged_df = naip_df_joined_dropna.merge(s2_df, on = "naip_grid", how = 'inner', suffixes = ['_naip','_s2'])
        merged_df.timestamp_naip = pd.to_datetime(merged_df.timestamp_naip)
        merged_df.timestamp_s2 = pd.to_datetime(merged_df.timestamp_s2)
        merged_df['time_diff'] = abs(merged_df.timestamp_naip - merged_df.timestamp_s2)
        # For non-unique keys, the pair with the smallest time difference is kept
        min_timediff = merged_df.groupby(by = ['UUID_naip','naip_grid']).time_diff.idxmin()
        paired_df = merged_df.loc[min_timediff].reset_index(drop=True)
        paired_df.drop('time_diff',axis = 1,inplace = True)


        # STEP 4: only keep the records with available landcover segmentation label
        label_obtained = []
        for index, row in paired_df.iterrows():
            label_mask_path = os.path.join(satlas_data_path, f'{split}_labels', row['s2_grid'], 'label_0','land_cover.png')
            label_obtained.append(os.path.exists(label_mask_path))
        paired_df = paired_df.loc[label_obtained]

        # STEP 5: save metadata of pairs to the new data path
        if not os.path.exists(usat_data_path):
            os.makedirs(usat_data_path)
        if split == 'train':
            paired_df.to_csv(os.path.join(usat_data_path,'paired_metadata_with_landcover.csv'), index = False)
        else:
            paired_df.to_csv(os.path.join(usat_data_path,'paired_metadata_val_with_landcover.csv'), index = False)

        print(f"finished writing {split} paired metadata")