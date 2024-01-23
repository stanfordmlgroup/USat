import csv
import typing as T
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def extract_model_state_dict_from_ckpt (ckpt: T.Dict[str, T.Any]):
    """Extract model state dict from ckpt.
    Note: the key of model_state_dict is the name of your
    model variable in the task.
    """
    state_dict = ckpt['state_dict']
    model_state_dict = {}
    for key, state in state_dict.items():
        key = key.split('.')
        model = key[0]
        layer = '.'.join(key[1:])
        if model not in model_state_dict:
            model_state_dict[model] = OrderedDict()
        model_state_dict[model][layer] = state.clone()
    return model_state_dict


def dict_list_to_csv(path: str, dict_list: T.List[T.Dict[str, T.Any]]) -> None:
    """Write a dictionary to CSV file.

    Keyword arguments:
        path: path to CSV file to be written to
        dict: dictionary whose values are to be stored
    """
    with open(path, 'w') as f:
        dict_writer = csv.DictWriter(f, dict_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)


def csv_to_dict_list(path: str) -> None:
    """Read a dictionary from a CSV file.

    Keyword arguments:
        path: path to CSV file to be read from
    """
    with open(path, 'r') as f:
        dict_reader = csv.DictReader(f)
        return list(dict_reader)


def compute_dataset_stats(dataset: Dataset, num_channels: int = 3) -> None:
    """ Compute the min, max, mean, and std of a given PyTorch compatible
    datset. Assumes that the data tensor is set as (C, H, W).
    """
    num_pixels = 0
    channels_sum = torch.zeros(num_channels)
    channels_squared_sum = torch.zeros(num_channels)

    dl = DataLoader(dataset, batch_size=256, num_workers=64)
    for _, (data, _) in enumerate(tqdm(dl)):
        channels_sum += torch.sum(data, (0,2,3))
        channels_squared_sum += torch.sum(data**2, (0,2,3))
        num_pixels += data.size(0) * data.size(2) * data.size(3)

    mean = channels_sum / num_pixels
    std = torch.sqrt((channels_squared_sum / num_pixels) - (mean ** 2))
    print(f"Dataset len: {len(dataset)}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")