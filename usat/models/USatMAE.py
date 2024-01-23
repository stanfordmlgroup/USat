import typing as T
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import  Block

from usat.utils.pos_embed import *
from usat.utils.builder import MODEL
from usat.models.USatViT import USatViTEncoder

class USatMAE(USatViTEncoder):
    def __init__(self, input_params: T.Dict[str, T.Any],
                       encode_type: str = 'concat', 
                       #----Encoder specific  
                       embed_dim: int = 1024, 
                       depth: int = 24, 
                       num_heads: int = 16, 
                       embed_band: bool = True,
                       band_embed_dim: int = 128,
                       pad_band_embed_null: bool = False,
                       embed_prod: bool = True,
                       prod_embed_dim: int = 128,
                       pad_prod_embed_null: bool = False,
                       #----Decoder specific
                       decoder_embed_dim: int = 512,
                       decoder_depth: int = 8,
                       decoder_num_heads: int = 16,
                       decoder_band_embed_dim: int = 64,
                       decoder_prod_embed_dim: int = 64,
                       #----Common
                       mlp_ratio: float =4., 
                       qkv_bias: bool = True,
                       norm_layer: nn.Module = nn.LayerNorm,
                       norm_pix_loss=False) -> None:
        #=======================================================================
        # MAE encoder specific from parent class
        super().__init__(input_params, encode_type, embed_dim, depth, num_heads, 
                         embed_prod, prod_embed_dim, pad_prod_embed_null, 
                         embed_band, band_embed_dim, pad_band_embed_null, 
                         mlp_ratio, qkv_bias, norm_layer)

        #=======================================================================
        # MAE defcoder specific
        # Initialize decoder projection
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Initialize mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Initialize decoder embedding
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_prod_embed_dim = decoder_prod_embed_dim if embed_prod else 0
        self.decoder_band_embed_dim = decoder_band_embed_dim if embed_band else 0
        self.decoder_pos_embed_dim = decoder_embed_dim - self.decoder_prod_embed_dim - self.decoder_band_embed_dim

        self.decoder_pos_embed, self.decoder_band_embed, self.decoder_prod_embed = self.initialize_embedding(self.decoder_pos_embed_dim, 
                                                                                                            self.decoder_band_embed_dim, 
                                                                                                            self.decoder_prod_embed_dim,
                                                                                                            sum_dim=self.decoder_embed_dim)

        # Initialize decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # TODO: might be different for different tasks (e.g. different output dimension)
        self.decode_pred = {}
        for group_name, group_membners in self.groups.items():
            params = self.input_params[group_membners[0]]
            kernel_size = int(self.ground_cover//float(params['GSD'])//params['num_patch'])
            self.decode_pred[group_name] = nn.Linear(decoder_embed_dim, len(group_membners) * kernel_size**2)
        self.decode_pred = nn.ModuleDict(self.decode_pred)

        self.norm_pix_loss = norm_pix_loss
        #=======================================================================
        # need to run this again (it has been run once in parent class initialization in super call) 
        #  to make sure all the parameters are initialized properly
        self.initialize_weight()

    
    def initialize_weight(self) -> None:
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)


    def get_auxilliary_embed(self, original_input: T.Dict[str, torch.Tensor], 
                                   encoder: bool = True) -> T.Union[T.Dict[str, torch.Tensor], torch.Tensor]:
        if encoder:
            return super().get_auxilliary_embed(original_input)
        else:
            auxilliary_embed = []
            if self.encode_type == 'concat':
                for group_name, group_member in self.groups.items():
                    product_band = group_member[0]
                    product, band = product_band.split(':')
                    auxilliary_embed.append(self.decoder_pos_embed[str(self.input_params[product_band]['GSD'])])
                    if self.embed_band:
                        # Expand to sequence length and concatenate
                        auxilliary_embed[-1] = torch.cat((auxilliary_embed[-1], self.decoder_band_embed[group_name].expand(auxilliary_embed[-1].shape[0], -1)), dim=-1)
                    if self.embed_prod:
                        # Expand to sequence length and concatenate
                        auxilliary_embed[-1] = torch.cat((auxilliary_embed[-1], self.decoder_prod_embed[product].expand(auxilliary_embed[-1].shape[0], -1)), dim=-1)
                    
                    data = original_input[product_band]
                    band_ground_cover = int(data.shape[-1] * self.input_params[product_band]['GSD'])
                    if band_ground_cover not in self.selection_index[group_name]:
                        raise ValueError(f'Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, ' +
                                        f'which does not fit in a concentric square grid of {self.input_params[product_band]["num_patch"]}x{self.input_params[product_band]["num_patch"]}' +
                                        f'patches for define ground cover {self.ground_cover}x{self.ground_cover}.')
                    auxilliary_embed[-1] = auxilliary_embed[-1][self.selection_index[group_name][band_ground_cover]]
            elif self.encode_type == "sum":
                for group_name, group_member in self.groups.items():
                    product_band = group_member[0]
                    product, band = product_band.split(':')
                    auxilliary_embed.append(self.decoder_pos_embed[str(self.input_params[product_band]['GSD'])])
                    if self.embed_band:
                        group_idx = int(group_name.split('group')[-1])
                        auxilliary_embed[-1] = auxilliary_embed[-1] + self.decoder_band_embed(torch.tensor(group_idx).to(original_input[product_band].device)).unsqueeze(0)
                    if self.embed_prod:
                        unique_prod = {prduct_band.split(':')[0]: None for prduct_band in self.input_params.keys()}
                        unique_prod = sorted(unique_prod.keys())
                        product_idx = unique_prod.index(product)
                        auxilliary_embed[-1] = auxilliary_embed[-1] + self.decoder_prod_embed(torch.tensor(product_idx).to(original_input[product_band].device)).unsqueeze(0)

                    data = original_input[product_band]
                    band_ground_cover = int(data.shape[-1] * self.input_params[product_band]['GSD'])
                    if band_ground_cover not in self.selection_index[group_name]:
                        raise ValueError(f'Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, ' +
                                        f'which does not fit in a concentric square grid of {self.input_params[product_band]["num_patch"]}x{self.input_params[product_band]["num_patch"]}' +
                                        f'patches for define ground cover {self.ground_cover}x{self.ground_cover}.')
                    auxilliary_embed[-1] = auxilliary_embed[-1][self.selection_index[group_name][band_ground_cover]]
            else:
                raise Exception("Invalid encode type")

            auxilliary_embed = torch.cat(auxilliary_embed, dim=0)
            return auxilliary_embed


    def random_masking(self, x: torch.Tensor, 
                             intervals: T.List[T.Tuple[int, int, bool, int]],
                             mask_ratio: float) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adapt from SatMAE: https://github.com/sustainlab-group/SatMAE

        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise. Modified to allow
        for per-band random masking to ensure that mask_ratio is consistent
        across bands with a different number of patches.
        x: [B, T, D], sequence
        """
        B, T, D = x.shape  # batch, sequence length, dim

        all_ids_restore, all_ids_keep, all_masks = [], [], []
        for interval in intervals:
            start, stop, needs_masking, _ = interval
            size = stop - start
            len_keep = int(size * (1 - mask_ratio))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                # keep the first subset
                ids_keep = ids_shuffle[:, :len_keep]
                mask[:, :len_keep] = 0
            else:
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                ids_keep = torch.arange(size, device=x.device).repeat(B, 1)
                mask[:, :] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            ids_keep += start
            all_ids_restore.append(ids_restore)
            all_ids_keep.append(ids_keep)
            all_masks.append(mask)

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


    def consistent_masking(self, x: torch.Tensor,
                                 intervals: T.List[T.Tuple[int, int, bool, int]],
                                 mask_ratio: float, type: str = "random") -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        all_ids_restore = [None] * len(intervals)
        all_ids_keep = [None] * len(intervals)
        all_masks = [None] * len(intervals)
        base_num_patches = None
        base_keep_ids = []

        # Sort by larget GSD and apply spatial masks to those first
        idxs = np.argsort(np.array(intervals)[:, 3])
        for idx_num, idx in enumerate(idxs):
            start, stop, needs_masking, num_patches = intervals[idx]
            size = stop - start
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking:
                if base_num_patches is None:
                    base_num_patches = num_patches
                    if type == "random":
                        # Setup the initial masking on the last number of patches
                        noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                        # sort noise for each sample
                        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                        ids_restore = torch.argsort(ids_shuffle, dim=1)
                        # keep the first subset
                        len_keep = int(size * (1 - mask_ratio))
                        ids_keep = ids_shuffle[:, :len_keep]
                        mask[:, :len_keep] = 0
                    elif type == "structured":
                        # Obtain a border around the image, make it randomly sized across batches
                        if num_patches > 2:
                            border = max(1, min(num_patches // 2 - 1, torch.normal(num_patches // 4, 1, size=(1,1)).int().item()))
                            all_idxs = torch.arange(size, device=x.device).reshape(num_patches, num_patches)
                            ids_keep = all_idxs[border:-border, border:-border].flatten().repeat(B, 1)
                            ids_restore = torch.arange(size, device=x.device).repeat(B, 1)

                            row_num, col_num = ids_keep.shape
                            idx0 = torch.arange(row_num).reshape(-1, 1).repeat(1, col_num).flatten()
                            idx1 = ids_keep.flatten()
                            mask[idx0, idx1] = 0

                    # Copy this base set of indices to all other bands
                    base_keep_ids = ids_keep.clone()
                else:
                    # Convert each of the base_keep_ids into ids for the current entry
                    ratio = int(num_patches / base_num_patches)
                    ids_keep = base_keep_ids.repeat(int(ratio * ratio), 1, 1).permute(1, 2, 0)
                    for idx_x in range(ratio):
                        for idx_y in range(ratio):
                            entry_idx = idx_x * ratio + idx_y
                            row_scale = torch.div(ids_keep[:, :, entry_idx], base_num_patches, rounding_mode='floor')
                            col_scale = ids_keep[:, :, entry_idx] % base_num_patches
                            ids_keep[:, :, entry_idx] = row_scale * ratio * num_patches + col_scale * ratio + idx_x * num_patches + idx_y
                    ids_keep = ids_keep.reshape(ids_keep.shape[0], -1)
                    ids_restore = torch.arange(size, device=x.device).repeat(B, 1)

                    # Convert ids_keep into indexes for mask
                    row_num, col_num = ids_keep.shape
                    idx0 = torch.arange(row_num).reshape(-1, 1).repeat(1, col_num).flatten()
                    idx1 = ids_keep.flatten()
                    mask[idx0, idx1] = 0
            else:
                ids_keep = torch.arange(size, device=x.device).repeat(B, 1)
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                mask[:, :] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            ids_keep += start
            all_ids_restore[idx] = ids_restore
            all_ids_keep[idx] = ids_keep
            all_masks[idx] = mask

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


    def full_masking(self, x, intervals):
        B, T, D = x.shape
        ids_restore = torch.arange(T, device=x.device).repeat(B, 1)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([B, T], device=x.device)
        for interval in intervals:
            start, stop, needs_masking, _ = interval
            if needs_masking:
                mask[:, start:stop] = 1
        mask_idxs = (mask[0,:] == 0).nonzero().repeat(B, 1, D)
        x_masked = torch.gather(x, dim=1, index=mask_idxs)
        return x_masked, mask, ids_restore


    def mixed_masking(self, x, intervals, mask_ratio):

        B, T, D = x.shape  # batch, sequence length, dim

        all_ids_restore, all_ids_keep, all_masks = [], [], []

        # Compute total number of patches
        num_patches = sum([stop - start for start, stop, needs_masking, _ in intervals if not needs_masking])

        for interval in intervals:
            start, stop, needs_masking, _ = interval
            size = stop - start

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, size], device=x.device)
            if needs_masking: 
                ids_restore = torch.arange(size, device=x.device).repeat(B, 1)
                ids_keep = None
                mask[:, :] = 1
            else:
                len_keep = int(size * (1 - mask_ratio))
                noise = torch.rand(B, size, device=x.device)  # noise in [0, 1]
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                # keep the first subset
                ids_keep = ids_shuffle[:, :len_keep]
                mask[:, :len_keep] = 0

            # Update the index to match real position in original tensor
            ids_restore += start
            all_ids_restore.append(ids_restore)
            if ids_keep is not None:
                ids_keep += start
                all_ids_keep.append(ids_keep)
            all_masks.append(mask)

        ids_restore = torch.cat(all_ids_restore, dim=1)
        ids_keep = torch.cat(all_ids_keep, dim=1)
        mask = torch.cat(all_masks, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


    def compute_band_metadata(self, x: torch.Tensor,
                                    group_embed: T.Dict[str, torch.Tensor],
                                    spectral_mask_ratio: float) -> T.OrderedDict[str, T.OrderedDict[str, T.Tuple[int, int, bool, int]]]:
        """ Compute per-product, per-band metadata - range of patch embed indices
        for a given band, whether or not spatial mask should be applied to a given
        band, and number of patches. This is computed every batch to support
        multi-dataset training schemes.
        Ex.
        metadata = {
            "S2": {
                "Red": (12, 34, True, 4),
                "Green": (34, 56, True, 4)
            },
            "NAIP": {
                "Red": (12, 34, False, 8),
                "Green": (34, 56, False, 8)
            }
        }
        """
        masked_groups = []
        if self.masking_schema:
            all_groups = list(group_embed.keys())
            # Specify the groups to mask by index
            if isinstance(self.masking_schema["spectral"], list):
                masked_groups.extend([all_groups[idx] for idx in self.masking_schema["spectral"]])
            # Specify predefined spectral masking pattern
            elif isinstance(self.masking_schema["spectral"], str):
                if self.masking_schema["spectral"] == "random_groups":
                    masked_groups = np.random.choice(all_groups, size=int(len(all_groups) * spectral_mask_ratio), replace=False)
                elif self.masking_schema["spectral"] == "one_group":
                    masked_groups = np.random.choice(all_groups)
                elif self.masking_schema["spectral"] == "all_but_one_group":
                    masked_groups = np.random.choice(all_groups, size=len(all_groups)-1, replace=False)
                elif self.masking_schema["spectral"] == "all":
                    masked_groups.extend(all_groups)
                else:
                    raise Exception("Invalid spectral masking scheme provided")

        running_idx = 0
        metadata = OrderedDict()
        for group, data in group_embed.items():
            num_patches = data.shape[1]
            incr = running_idx + num_patches
            if group in masked_groups:
                metadata[group] = (running_idx, incr, True, num_patches ** 0.5)
            else:
                metadata[group] = (running_idx, incr, False, num_patches ** 0.5)
            running_idx = incr

        return metadata
    
    def mask(self, x: torch.Tensor,
                   band_metadata: T.OrderedDict[str, T.OrderedDict[str, T.Tuple[int, int, bool, int]]],
                   mask_ratio: float) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.masking_schema is None:
            return self.random_masking(x, [(0, x.shape[1], True, 0)], mask_ratio)

        interval_list = list(band_metadata.values())
        if self.masking_schema["spatial"] == "random":
            return self.random_masking(x, [(0, x.shape[1], True, 0)], mask_ratio)
        elif self.masking_schema["spatial"] == "random_same_ratio":
            return self.random_masking(x, interval_list, mask_ratio)
        elif self.masking_schema["spatial"] == "random_consistent":
            return self.consistent_masking(x, interval_list, mask_ratio, type="random")
        elif self.masking_schema["spatial"] == "structured":
            return self.consistent_masking(x, interval_list, mask_ratio, type="structured")
        elif self.masking_schema["spatial"] == "all":
            return self.full_masking(x, interval_list)
        elif self.masking_schema["spatial"] == "mixed":
            return self.mixed_masking(x, interval_list, mask_ratio)
        else:
            raise Exception("Invalid spatial masking scheme")
    

    def forward_encoder(self, x: T.Dict[str, torch.Tensor], 
                        mask_ratio: float, spectral_mask_ratio: float) -> torch.Tensor:
        # overwrite to add random masking
        # B: Batch size, H: Height, W: Width, p: patch size
        # D: Embedding dimension, N_n: Number of patches for each product_band
        # T: Sequence length (number of patches * number of spectral bands)
        # r: mask ratio
        # x = {'product:band': (B, 1, H, W), ...}
        # extract patche#s
        patch_embed = self.ind_patch_embed(x)   #{'product:band': (B, N_n, D), ...}

        group_embed = self.aggregate_by_group(patch_embed) # {'group0': (B, N_n, D), ...}
        #group_embed = self.group_patch_embed(x)

        # add additional embedding
        auxilliary_embed = self.get_auxilliary_embed(x, True) # {'group0': (N_n, D), ...}

        # compute metadata before concat bands
        band_metadata = self.compute_band_metadata(x, group_embed, spectral_mask_ratio)

        # Determine the bands + products you want to apply masking to
        x = torch.cat([group_embed[group_name].add_(auxilliary_embed[group_name]) for group_name in group_embed.keys()], dim=1) # (B, T, D)

        # Apply mask
        x, mask, ids_restore = self.mask(x, band_metadata, mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, (1-r)*T + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    

    def forward_decoder(self, laten: torch.Tensor, 
                              ids_restore: torch.Tensor,
                              mask: torch.Tensor,
                              original_input: T.Dict[str, torch.Tensor]) -> T.Tuple[T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor]]:
        # dD: decoder embedding dimension
        # project to decoder dimension
        x = self.decoder_embed(laten)  # (B, (1-r)*T + 1, dD)


        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # expand to (B, T*r, dD)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (B, 1 + T, D)

        auxilliary_embed = self.get_auxilliary_embed(original_input, False) # [T, dD]
        x[:, 1:, :].add_(auxilliary_embed) # add additional embedding

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :] # (N, T, dD)

        start_idx = 0
        out = {}
        mask_out = {}
        # Important that we iterate through this in the same order we encoded
        for group_name, group_member in self.groups.items():
            product_band = group_member[0]
            band_ground_cover = int(original_input[product_band].shape[-1] * self.input_params[product_band]['GSD'])
            num_patches = len(self.selection_index[group_name][band_ground_cover][0])
            # (B * N_n , dD)
            x_ = x[:, start_idx:start_idx+num_patches, :].reshape(-1, x.shape[-1])
            # {(B, N_n, len(groups),  p**2), ...}
            decoded_group_out = self.decode_pred[group_name](x_).reshape(x.shape[0], num_patches, len(group_member), -1)
            for idx, product_band in enumerate(group_member):
                out[product_band] = decoded_group_out[:, :, idx, :]
                # same mask for all bands in the same group
                mask_out[product_band] = mask[:, start_idx:start_idx+num_patches]
            start_idx += num_patches
            
        return out, mask_out
    

    def reconstruction_loss(self, 
                            input_dict: T.Dict[str, torch.Tensor], 
                            pred_dict: T.Dict[str, torch.Tensor], 
                            mask_dict: T.Dict[str, torch.Tensor]):
        """
        input_dict: {'product:band': (B, 1, H, W), ...}
        pred_dict: {'product:band': (B, N_n, p**2), ...}
        mask: {'product:band': (B, N_n), ...}

        Vanilla implementation of reconstruction loss, we reconstruct every original spectral bands,
        could make change later to something else.
        """
        total_loss = 0
        num_remove = 0
        for product_band in input_dict:
            spectral_img = input_dict[product_band]
            pred = pred_dict[product_band]
            mask = mask_dict[product_band]
            patch_size = int(self.ground_cover//float(self.input_params[product_band]['GSD'])//self.input_params[product_band]['num_patch'])
            
            target = self.patchify(spectral_img, patch_size, 1) # [B, N_n, p**2]

            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [B, N_n], mean loss per patch

            total_loss += (loss * mask).sum()
            num_remove += mask.sum()  # mean loss on removed patches
        return total_loss / num_remove


    def forward(self, x: T.Dict[str, torch.Tensor], 
                      mask_ratio: float, spectral_mask_ratio: float) -> T.Tuple[T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor]]:
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, spectral_mask_ratio)
        pred, mask = self.forward_decoder(latent, ids_restore, mask, x)
        loss = self.reconstruction_loss(x, pred, mask)
        return loss, pred, mask
    

    def patchify(self, imgs, p, c=1):
        """
        imgs: (B, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (B, L, C*patch_size**2)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x


    def unpatchify(self, x, p, c=1):
        """
        x: (B, L, C*patch_size**2)
        p: Patch embed patch size
        c: Num channels
        imgs: (B, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


@MODEL.register_module()
def usat_mae_small(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=384, depth=12, num_heads=6,
                        embed_band=True, band_embed_dim=32,
                        embed_prod=True, prod_embed_dim=32,
                        decoder_embed_dim = 192,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 16,
                        decoder_prod_embed_dim = 16,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_base(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=768, depth=12, num_heads=12,
                        embed_band=True, band_embed_dim=64,
                        embed_prod=True, prod_embed_dim=64,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)

@MODEL.register_module()
def usat_mae_large_only_pos_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=False, band_embed_dim=0,
                        embed_prod=False, prod_embed_dim=0,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large_only_pos_embed_pad_null(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128, pad_band_embed_null=True,
                        embed_prod=True, prod_embed_dim=128, pad_prod_embed_null=True,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large_reweighted(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=252,
                        embed_prod=True, prod_embed_dim=4,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)



@MODEL.register_module()
def usat_mae_large_no_prod_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=256,
                        embed_prod=False, prod_embed_dim=0,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 128,
                        decoder_prod_embed_dim = 0,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large_no_group_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=False, band_embed_dim=0,
                        embed_prod=True, prod_embed_dim=256,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 0,
                        decoder_prod_embed_dim = 128,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large_prod_embed_pad_null(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128, pad_prod_embed_null=True,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 128,
                        decoder_prod_embed_dim = 0,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)



@MODEL.register_module()
def usat_mae_large_384(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1152, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=256,
                        embed_prod=True, prod_embed_dim=128,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_large_384_no_prod_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1152, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=384,
                        embed_prod=False, prod_embed_dim=0,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)


@MODEL.register_module()
def usat_mae_huge(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1280, depth=32, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128,
                        decoder_embed_dim = 512,
                        decoder_depth = 8, decoder_num_heads = 16,
                        decoder_band_embed_dim = 64,
                        decoder_prod_embed_dim = 64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatMAE(**model_kwargs)
