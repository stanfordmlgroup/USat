import typing as T

import torch
import torchvision.transforms as transforms
import numpy as np

from usat.utils.builder import PRETRAIN_TRANSFORM



@PRETRAIN_TRANSFORM.register_module()
class RandomContrast:
    def __init__(self, min_c: float = 1, 
                       max_c: float = 1, 
                       clamp: T.Tuple[float] = None) -> None:
        self.min_c = min_c
        self.max_c = max_c
        self.clamp = clamp


    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        contrast_factor = (self.max_c-self.min_c)*torch.rand(1)+self.min_c
        contrast_factor = contrast_factor.type_as(scene)
        
        mean = torch.mean(scene)
        adjusted = (scene-mean)*contrast_factor+mean

        if self.clamp is not None:
            adjusted = torch.clamp(adjusted, min=self.clamp[0], max=self.clamp[0])

        return adjusted


@PRETRAIN_TRANSFORM.register_module()
class RandomBrightness:
    def __init__(self, max_b: float = 0,
                       clamp: T.Tuple[float] = None) -> None:
        self.max_b = max_b
        self.clamp = clamp

    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        brightness_factor = self.max_b*2*torch.rand(1)-self.max_b
        brightness_factor = brightness_factor.type_as(scene)

        if self.clamp is not None:
            adjusted = torch.clamp(adjusted, min=self.clamp[0], max=self.clamp[0])

        return adjusted


@PRETRAIN_TRANSFORM.register_module()
class ProductDropout:
    def __init__(self, group: T.List[int], 
                       drop_rate: float, 
                       training: bool) -> None:
        self.group = group
        self.drop_rate = drop_rate
        self.training = training
        self.num_group = len(self.group)
        self.group_index = self.group.cumsum(0)


    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        if scene.shape[1] != sum(self.group):
            raise ValueError('Number of group {sum(group)} not equal to number of channel {a.shape[1]}')

        out = None
        if self.training:
            mask = torch.zeros_like(scene)
            rand_group = np.random.rand(self.num_group)
            prod_to_drop = rand_group <= self.drop_rate
            prod_to_keep = np.random.choice(self.num_group, 1)
            prod_kept = 0
            for i in range(self.num_group):
                if i == 0:
                    start_index = 0
                else:
                    start_index = self.group_index[i-1]
                end_index = self.group_index[i]

                if prod_to_keep != i and prod_to_drop[i]:
                    continue
                
                mask[:, start_index:end_index, ...] = 1.0
                prod_kept += self.group[i]

            out = mask * scene / (prod_kept/scene.shape[1])

        else:
            out = scene
        
        return out


@PRETRAIN_TRANSFORM.register_module()
class RandomFlip:
    def __init__(self, flip_ud: bool, 
                       flip_lr: bool) -> None:
        """Assuming Last two dimension is [H, W]
        """
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr

    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        if self.flip_ud:
            ud_coin = torch.rand(1)
            if ud_coin > 0.5:
                scene = torch.flip(scene, dims=[-2])

        if self.flip_lr:
            lr_coin = torch.rand(1)
            if lr_coin > 0.5:
                scene = torch.flip(scene, dims=[-1])
        
        return scene


@PRETRAIN_TRANSFORM.register_module()
class RandomRotate:
    def __init__(self, group: T.List[int] = None) -> None:
        """Assuming Last two dimension is [H, W]
        """
        self.group = group

    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        if self.group is not None:
            pass
        k = torch.randint(0,4,(1,))
        scene = torch.rot90(scene, int(k), [-2,-1])

        return scene


@PRETRAIN_TRANSFORM.register_module()
def RGBColorJitter(brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
    """pass through for Color Jitter only work with RGB image"""
    # TODO: create a channel agnostic version of color Jitter
    return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)


@PRETRAIN_TRANSFORM.register_module()
def GaussianBlur(kernel_size: T.Union[int, T.Sequence], sigma: T.Tuple[float] = [0.1, 2.0]):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)


@PRETRAIN_TRANSFORM.register_module()
def RGBRandomSolarize(threshold: float, p: float = 0.5):
    return transforms.RandomSolarize(threshold, p=p)

@PRETRAIN_TRANSFORM.register_module()
def RandomResizedCrop(size: T.Union[int, T.Sequence], 
                      scale: T.Tuple[float] = (0.08, 1.0), 
                      ratio: T.Tuple[float] =(3.0 / 4.0, 4.0 / 3.0)):
    return transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)


@PRETRAIN_TRANSFORM.register_module()
def Normalize(mean: T.Sequence, 
              std: T.Sequence, 
              inplace: bool=False):
    return transforms.Normalize(mean=mean, std=std, inplace=inplace)


@PRETRAIN_TRANSFORM.register_module()
class RGBGaussianHighPass:
    def __init__(self, sigma: int = 5,
                       kernel: T.Tuple[int, int] = (9, 9)) -> None:
        self.sigma = sigma
        self.kernel = kernel
        self.gaussian_blur = transforms.GaussianBlur(kernel, (sigma, sigma))


    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        lowpass = self.gaussian_blur(scene)
        gauss_highpass = scene - lowpass + 0.5

        return gauss_highpass


@PRETRAIN_TRANSFORM.register_module()
class ImagePatchMask:
    def __init__(self, img_size: int = 224,
                       patch_per_side: int = 7,
                       grid_mask_p: float = 0.7,
                       independent_mask: bool = True,
                       gaussian_mask: bool = True,) -> None:

        # TODO: Currently only working with square images. Extend to arbitary aspect ratio later?
        # or making it working for any image size
        self.img_size = img_size
        self.grid_size = patch_per_side

        self.grid_mask_p = grid_mask_p
        self.independent_mask = independent_mask
        self.gaussian_mask = gaussian_mask
        
        self.num_patch = patch_per_side * patch_per_side

        if img_size % patch_per_side != 0:
            raise ValueError(f'Image size must be divisible by patch per side.')

        self.patch_size = img_size // patch_per_side



    def __call__(self, scene: torch.Tensor) -> torch.Tensor:
        if scene.dim() == 3:
            num_channel, w, h = scene.shape
            batch_size = 1
        elif scene.dim() == 4:
            batch_size, num_channel, w, h = scene.shape
        
        assert w == self.img_size, f'Width of the image {w} does not match image size: {self.img_size}'
        assert h == self.img_size, f'Width of the image {h} does not match image size: {self.img_size}'

        grid_mask = torch.ones(batch_size, num_channel, self.img_size, self.img_size, device=scene.device)

        patches = grid_mask.unfold(2, self.patch_size, self.patch_size).unfold(num_channel, self.patch_size, self.patch_size).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.num_patch, 3, self.patch_size, self.patch_size)

        if self.independent_mask:
            mask_idx = torch.rand(batch_size, self.num_patch, num_channel, device=scene.device) < self.grid_mask_p
            patches[mask_idx] = 0
        else:
            mask_idx = torch.rand(batch_size, self.num_patch, device=scene.device) < self.grid_mask_p
            patches[mask_idx] = 0

        patches_reorg = patches.permute(0,2,3,4,1).reshape(batch_size, -1, self.num_patch)
        grid_mask = torch.nn.functional.fold(patches_reorg, self.img_size, kernel_size=self.patch_size, stride=self.patch_size)
        grid_mask = grid_mask == 1

        if self.gaussian_mask:
            masked_image = torch.where(grid_mask[0], scene, torch.randn_like(scene, device=scene.device, dtype=scene.dtype)*0.1+0.5)
        else:
            masked_image = torch.where(grid_mask[0], scene, torch.zeros_like(scene, device=scene.device, dtype=scene.dtype)+0.5)

        return masked_image


