import typing as T

import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from usat.utils.builder import PRETRAIN_TRANSFORM
from usat.core.serialization import read_yaml

def visualize_RGB_tensor (tensor: torch.Tensor, bound: T.Tuple[int] = None):
    if tensor.dim() == 3:
        tensor = tensor[None, ...]

    if tensor.shape[1] != 3 and tensor.shape[1] != 1:
        raise ValueError(f'Channel dimension of the RGB tensor must be 3 or 1 but get {tensor.shape[1]}')
    
    np_imgs = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    num_img = np_imgs.shape[0]

    fig = plt.figure(figsize=(5, num_img*5))
    for i, np_img in enumerate(np_imgs):
        if bound is not None:
            img_min, img_max = bound
        else:
            img_min = np.min(np_img)
            img_max = np.max(np_img)
        
        np_img_norm = (np_img - img_min) / (img_max - img_min)
        
        ax = fig.add_subplot(num_img, 1, i+1)
        ax.imshow(np_img_norm)
        ax.axis('off')



def visualize_transform (dataset: Dataset, 
                         cfg: str, 
                         transform_names: T.List[str], 
                         n_sample: int,
                         random: bool = False,
                         channel_group: T.List[T.List[int]] = [[0, 1, 2]], 
                         bound: T.List[T.List[int]] = [[0, 1]]):
    """Visualize transformation on a dataset

    Args:
        dataset (Dataset): An initialized pytorch dataset
        cfg (str): path to cfg file
        transform_name (T.List[str]): Name of the transform in the cfg file
        n_sample (int): number of sample to visualize 
        random (bool, optional): If you want to visualize random sample or first n sample. Defaults to False.
        channel_group (T.List[T.List[int]], optional): Group together the channel to visualize, for example [[0,1,2],[3]] will treat first 
            3 channel as RGB and last channel as gray scale. Defaults to [[0, 1, 2]].
        bound (T.List[T.List[int]], optional): Upper and lower limit of the number, should have same length as channel_group. 
            Defaults to [[0, 1]].
    """
    if len(channel_group) != len(bound):
        raise ValueError(f'Length of channel_group is {len(channel_group)}, but only have {len(bound)} bound')
    
    n_group = len(channel_group)
    n_data = len(dataset)

    if random:
        idx = np.random.choice(n_data,  n_sample, replace=False)  
    else:
        idx = np.arange(n_sample)    

    cfg = read_yaml(cfg)
    transforms = [PRETRAIN_TRANSFORM.build(cfg, target=name) for name in transform_names]
    for t in transforms:
        # remove normalize layer as it is difficult to visualize
        if isinstance(t.transforms[-1], Normalize):
            t.transforms = t.transforms[:-1]
    n_transform = len(transforms)

    fig = plt.figure(figsize=(2 * (n_transform + 1) * n_group, 2 * n_sample))
    col = (n_transform + 1) * n_group
    row = n_sample
    n_ax = 1

    for i in idx:
        x = dataset[i][0]
        full_np_imgs = x.permute(1, 2, 0).detach().cpu().numpy()
        transform_imgs = [t(x) for t in transforms]
        for group_idx, group_bound in zip(channel_group, bound):
            img_min, img_max = group_bound

            np_img = full_np_imgs[:, :, group_idx]

            ax = fig.add_subplot(row, col, n_ax)
            ax.imshow(np_img)
            ax.axis('off')
            ax.set_title('original')
            n_ax += 1

            for j, transform_img in enumerate(transform_imgs):
                np_img = transform_img[group_idx, :, :].permute(1, 2, 0).detach().cpu().numpy()
                np_img_norm = (np_img - img_min) / (img_max - img_min)

                ax = fig.add_subplot(row, col, n_ax)
                ax.imshow(np_img_norm)
                ax.axis('off')
                ax.set_title(f'{transform_names[j]}')
                n_ax += 1
    plt.tight_layout()