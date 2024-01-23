GENERAL_DATA_DIR = "/PATH/TO/GENERAL/DATA/DIR"
WANDB_ENTITY = "aicc"
WANDB_PROJECT = "self-sup"

NAIP_MEAN = [0.48297,0.47841,0.42605,0.60955]
NAIP_STD = [0.17642,0.13683,0.11934,0.13406]

FMOW_RGB_MEAN = [0.4275, 0.4359, 0.4052]
FMOW_RGB_STD = [0.1925, 0.1840, 0.1842]

IN_RGB_MEAN = [0.485, 0.456, 0.406]
IN_RGB_STD = [0.229, 0.224, 0.225]

# Statistics based on TG 269695 sample train dataset (TG excludes bad patches)
# using images rescaled to 224x224 for simplicity
BEN_TG_TRAIN_MEAN = [353.2649, 441.8239, 625.1196, 602.2891, 961.2896,
                     1795.3566, 2075.0393, 2218.1025, 2264.2021, 2244.3267,
                     1585.2673, 1005.3627]
BEN_TG_TRAIN_STD = [353.2649, 441.8239, 625.1196, 602.2891, 961.2896, 1795.3566,
                    2075.0393, 2218.1025, 2264.2021, 2244.3267, 1585.2673,
                    1005.3627]
BEN_TG_SCALED_TRAIN_MEAN = [0.4560, 0.4696, 0.4838, 0.4791, 0.4913, 0.4969,
                            0.4975, 0.4975, 0.4978, 0.4979, 0.4978, 0.4955]
BEN_TG_SCALED_TRAIN_STD = [0.1759, 0.1658, 0.1552, 0.1981, 0.1576, 0.1477,
                           0.1506, 0.1515, 0.1478, 0.1434, 0.1688, 0.1944]

# Statistics based on TF 311667 sample train dataset excluding bad patches
# using images rescaled to 224x224 for simplicity
BEN_TF_NO_BAD_PATCH_TRAIN_MEAN = [339.9331, 428.9947, 613.1380, 589.3089, 
                                  949.3361, 1789.6952, 2072.3052, 2215.6262,
                                  2263.3401, 2243.1606, 1594.0305, 1008.6794]
BEN_TF_NO_BAD_PATCH_TRAIN_STD = [541.6486, 564.4072, 574.8396, 667.4680,
                                 718.4330, 1082.9532, 1259.0577, 1354.4314,
                                 1342.2411, 1285.6140, 1074.2240,  812.0288]
BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_MEAN = [0.4796, 0.4821, 0.4844, 0.4852, 0.4877,
                                         0.4940, 0.4944, 0.4947, 0.4953, 0.4960,
                                         0.4941, 0.4907]
BEN_TF_NO_BAD_PATCH_SCALED_TRAIN_STD = [0.1304, 0.1368, 0.1624, 0.1813, 0.1991,
                                        0.2362, 0.2398, 0.2404, 0.2426, 0.2445,
                                        0.2402, 0.2304]
BEN_S1_TF_TRAIN_MEAN = [-19.2485, -12.6027]
BEN_S1_TF_TRAIN_STD = [5.3695, 5.0129]
BEN_S1_TF_SCALED_TRAIN_MEAN = [0.5025, 0.5090]
BEN_S1_TF_SCALED_TRAIN_STD = [0.2397, 0.2171]


# STATISTICS FOR EUROSAT 
EUROSAT_CHANNEL_MEANS = [1354.4056, 1118.2439, 1042.9297,  947.6261, 1199.4731, 
                        1999.7911, 2369.2234, 2296.8264, 732.0834, 12.1133, 
                        1819.0103, 1118.9238, 2594.1406]
EUROSAT_CHANNEL_STD_DEVS = [ 245.7163,  333.0081,  395.0927,  593.7505,  566.4164,  861.1832,
                            1086.6305, 1117.9807,  404.9199,    4.7758, 1002.5875,  761.3034,
                            1231.5856]

# EuroSAT stats computed from train set. This is ordered from smallest to largest
# numerically. (B01, B02, ..., B11, B12)
EUROSAT_MEANS = [1354.4055, 1118.2440, 1042.9299,  947.6262, 1199.4725, 1999.7910,
                 2369.2239, 2296.8262,  732.0834,   12.1133, 1819.0100, 1118.9242,
                 2594.1409]
EUROSAT_STDS = [245.7171, 333.0075, 395.0922, 593.7505, 566.4178, 861.1844,
                1086.6289, 1117.9821, 404.9200, 4.7758, 1002.5884, 761.3026,
                1231.5856]

EUROSAT_SCALED_MEANS = [0.4926, 0.4896, 0.4900, 0.4908, 0.4933, 0.4994, 0.4982, 0.4983, 0.4947,
                        0.4892, 0.4975, 0.4954, 0.4986]
EUROSAT_SCALED_STSDS = [0.2265, 0.2081, 0.2132, 0.2207, 0.2299, 0.2444, 0.2446, 0.2457, 0.2362,
                        0.1948, 0.2438, 0.2373, 0.2463]

SENTINEL_1_BANDS = ["VV", "VH"]

SENTINEL_2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                    "B08", "B8A", "B09", "B10", "B11", "B12"]
 
IN_RESENET_CHANNEL_MAP = {
    'red': 0,
    'green': 1,
    'blue': 2
}

PROBING = "probing"
FINETUNING = "finetuning"
TRAIN_METHODS = [PROBING, FINETUNING]
