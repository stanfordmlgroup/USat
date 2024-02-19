Download the beta version of Satlas images (NAIP and Sentinel-2) and labels:
https://github.com/allenai/satlas/blob/3132ad62dae50f18306f777bbbb32c96a82ab4c8/README.md

Structure of the downloaded Satlas data:
```{r}
SATLAS_DATA_PATH/
    train/
        images/
            UUID/
                tci/
                    zoom_level_13_grid.png
                    ...
                b05/
                b06/
                ...
            ...
        highres/
            UUID/
                tci/
                    zoom_level_17_grid.png
            ...
    val/
        ...
    train_labels/
        zoom_level_13_grid/
            label_0/
                vector.json
            airplane_325/
    val_labels/
        ...
```


Structure of the processed USat Satlas data:
```{r}
USAT_DATA_PATH/
    train/
        s2_cropped_resized/
            000b0417499f43ffb8907b110d9793d6/
                tci/
                    1867_3287.png
                    1867_3288.png
                    ...
                b05/
                b06/
                ...
            000d5719cfbb4590aa9156fd49d69806/
                ...
            ...
        highres_resized/
            ...
        land_cover_cropped_resized/
    val/
        ...
    paired_metadata_multilabel.csv
    paired_metadata_val_multilabel.csv
```

To run the data preparation code, change the top three variables in run_satlas_prep.sh to be the input and destination path, and number of workers for multi-processing. 
