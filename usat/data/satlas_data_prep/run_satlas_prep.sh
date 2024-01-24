#!/bin/bash

SATLAS_DATA_PATH="/deep2/group/aicc-bootcamp/self-sup/satlas"
USAT_DATA_PATH="/scr/usat_retest"

# write paired_metadata_with_landcover.csv and paired_metadata_val_with_landcover.csv
python 1_write_metadata.py "$SATLAS_DATA_PATH" "$USAT_DATA_PATH"

# save the cropped and resized images
python 2_preprocess_images.py "$SATLAS_DATA_PATH" "$USAT_DATA_PATH" 90

# append multi-label classification column to metadata
python 3_write_multilabel.py "$SATLAS_DATA_PATH" "$USAT_DATA_PATH" 90