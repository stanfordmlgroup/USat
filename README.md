# USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery

> [**USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery**](TBD)<br>
> [Jeremy Irvin*](jirvin16.github.io), [Lucas Tao*](https://www.linkedin.com/in/lucas-tao-612026134), [Joanne Zhou](https://www.linkedin.com/in/joanneyoushanzhou), [Yuntao Ma](https://www.linkedin.com/in/yuntaoma0402), [Langston Nashold](https://www.linkedin.com/in/langston-nashold), [Benjamin Liu](https://profiles.stanford.edu/benjamin-liu), [Andrew Y. Ng](https://www.andrewng.org/)

*Equal contribution

Official implementation of the paper "[USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery](TBD)".

## Highlights
TBD

## Data Downloading
### USatlas

### METER-ML
The METER-ML dataset can be downloaded here: https://stanfordmlgroup.github.io/projects/meter-ml/. After downloading,
structure the dataset as follows:
- top_level_dir/
    - README
    - train_dataset.geojson
    - train_images/
        - train_images_1/
        - train_images_2/
        - train_images_3/
    - val_dataset.geojson
    - val_images/
    - test_dataset.geojson
    - test_images/

### EuroSAT
The EuroSAT dataset can be found here: https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat. The
dataset is downloaded automatically when 'download=True' in the config.

### BigEarthNet
The BigEarthNet (BEN) dataset can be found here: https://torchgeo.readthedocs.io/en/latest/api/datasets.html#bigearthnet. The
dataset is downloaded automatically when 'download=True' in the config.


## Environment Set-up
Performing the following step in `USat` directory. 
1. Create a Python 3.9 conda enviroment.
```bash
conda create -n usat python=3.9
```
2. Install dependency
```bash
pip install -r requirements.txt
```
3. Editable install the project code.
```bash
pip install --no-deps -e ./
```

## Data Preprocessing
TBD

## Model Training
### Pretrain USat on Satlas from Scratch
```bash
python main.py train usat/configs/pretrain/usat.yaml
```

### METER-ML Finetune
Uncomment and replace the model "ckpt:" line in `usat/configs/downstream/usat_meter_s2.yaml` with
the pretraind checkpoint of your choice. Then run the following:
```bash
python main.py train usat/configs/downstream/usat_meter_s2.yaml
```
Uncomment and replace the model "ckpt:" line in `usat/configs/downstream/usat_meter_naip.yaml` with
the pretraind checkpoint of your choice. Then run the following:
```bash
python main.py train usat/configs/downstream/usat_meter_naip.yaml
```

### EuroSAT
Uncomment and replace the model "ckpt:" line in `usat/configs/downstream/usat_eurosat.yaml` with
the pretrained checkpoint of your choice. Then run the following:
```bash
python main.py train usat/configs/downstream/usat_eurosat.yaml
```

### BigEarthNet
Uncomment and replace the model "ckpt:" line in `usat/configs/downstream/usat_ben.yaml` with
the pretrained checkpoint of your choice. Then run the following:
```bash
python main.py train usat/configs/downstream/usat_ben.yaml
```

## Model Evaluation

### METER-ML
Uncomment and replace the "test_ckpt:" line in `usat/configs/downstream/usat_meter_s2.yaml` with
the finetuned checkpoint to test. Then run the following (first for S2 and then for NAIP):
```bash
python main.py test usat/configs/downstream/usat_meter_s2.yaml
```

Uncomment and replace the "test_ckpt:" line in `usat/configs/downstream/usat_meter_naip.yaml` with
the finetuned checkpoint to test. Then run the following:
```bash
python main.py test usat/configs/downstream/usat_meter_naip.yaml
```

### EuroSAT
Uncomment and replace the "test_ckpt:" line in `usat/configs/downstream/usat_ben.yaml` with
the finetuned checkpoint to test. Then run the following:
```bash
python main.py test usat/configs/downstream/usat_eurosat.yaml
```

### BigEarthNet
Uncomment and replace the "test_ckpt:" line in `usat/configs/downstream/usat_ben.yaml` with
the finetuned checkpoint to test. Then run the following:
```bash
python main.py test usat/configs/downstream/usat_ben.yaml
```


## Citation
TBD

## Contact
TBD
