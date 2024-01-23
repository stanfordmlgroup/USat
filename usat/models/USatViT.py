from collections import defaultdict
import typing as T
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import  Block

from usat.utils.pos_embed import *
from usat.utils.builder import MODEL

# sample input params
# TODO: delete this later
input_params = {
    'ground_cover': 960,
    'P1:R': {'GSD': 1, 'num_patch': 16},
    'P1:G': {'GSD': 1, 'num_patch': 16},
    'P1:B': {'GSD': 1, 'num_patch': 16},
    'P2:R': {'GSD': 10, 'num_patch': 8},
    'P2:G': {'GSD': 10, 'num_patch': 8},
    'P2:B': {'GSD': 10, 'num_patch': 8},
    'P2:nir': {'GSD': 20, 'num_patch': 4},
}


class USatViTEncoder(nn.Module):
    def __init__(self, input_params: T.Dict[str, T.Any],
                       encode_type: str = 'concat', 
                       embed_dim: int = 1024, 
                       depth: int = 24, 
                       num_heads: int = 16, 
                       embed_prod: bool = True, # Product embedding for different product
                       prod_embed_dim: int = 128,
                       pad_prod_embed_null: bool = False,
                       embed_band: bool = True, # Band embedding for different bands
                       band_embed_dim: int = 128,
                       pad_band_embed_null: bool = False,
                       mlp_ratio: float =4., 
                       qkv_bias: bool = True,
                       norm_layer: nn.Module = nn.LayerNorm,) -> None:
        super().__init__()

        # Parse input params
        self.ground_cover = input_params.pop('ground_cover')
        self.masking_schema = input_params.pop('masking_schema', None)
        self.aggr_type = input_params.pop('aggr_type', 'mean')
        self.use_superposition_encoding = input_params.pop('use_superposition_encoding', True)
        self.overlap_factor = input_params.pop('overlap_factor', 1.0)
        self.input_params = input_params
        self.groups = self.validate_group(input_params.pop('groups', None)) # {'group0':[product_band, ...], 'group1': ..., ...}
        self.ind_patch_embed = IndPatchEmbed(self.ground_cover, input_params, embed_dim=embed_dim)
        #self.group_patch_embed = GroupPatchEmbed(self.ground_cover, input_params, embed_dim=embed_dim, groups=self.groups)

        # encoding type either to be concat of sum
        assert encode_type in ['concat', 'sum'], f'encode_type {encode_type} is not supported.'
        self.encode_type = encode_type
        
        # Initialize embedding
        self.embed_dim = embed_dim
        self.embed_prod = embed_prod
        self.prod_embed_dim = prod_embed_dim if embed_prod else 0
        self.pad_prod_embed_null = pad_prod_embed_null
        self.embed_band = embed_band
        self.band_embed_dim = band_embed_dim if embed_band else 0
        self.pad_band_embed_null = pad_band_embed_null
        self.pos_embed_dim = embed_dim - prod_embed_dim - band_embed_dim
        
        # NOTE: band_embed is now spectral group embed, not changing the name to avoid unexpected break
        self.pos_embed, self.band_embed, self.prod_embed = self.initialize_embedding(self.pos_embed_dim, self.band_embed_dim, self.prod_embed_dim, sum_dim=self.embed_dim)
        self.selection_index = self.initialize_selection_index()

        # Initialize cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()


    def validate_group (self, groups):
        valid_groups = {}
        if groups is None:
            for group_idx, prodcut_band in enumerate(self.input_params.keys()):
                valid_groups[f'group{group_idx}'] = [prodcut_band]
            return valid_groups
        for group_idx, group in enumerate(groups):
            group_product = None
            group_gsd = None
            group_num_patch = None
            valid_groups[f'group{group_idx}'] = []
            for product_band in group:
                product, _ = product_band.split(':')
                group_product = product if group_product is None else group_product
                group_gsd = self.input_params[product_band]['GSD'] if group_gsd is None else group_gsd
                group_num_patch = self.input_params[product_band]['num_patch'] if group_num_patch is None else group_num_patch
                if product != group_product:
                     raise ValueError(f'Product {product} in group {group} does not match with product {group_product} in the same group.')
                if self.input_params[product_band]['GSD'] != group_gsd:
                    raise ValueError(f'GSD {self.input_params[product_band]["GSD"]} in group {group} does not match with GSD {group_gsd} in the same group.')
                if self.input_params[product_band]['num_patch'] != group_num_patch:
                    raise ValueError(f'Number of patch {self.input_params[product_band]["num_patch"]} in group {group} does not match with number of patch {group_num_patch} in the same group.')
                valid_groups[f'group{group_idx}'].append(product_band)
        return valid_groups
    

    def initialize_embedding(self, pos_embed_dim: int, band_embed_dim: int, prod_embed_dim: int, sum_dim: int=0) -> T.Tuple[T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor]]:
        if self.encode_type == 'sum':
            return self.initialize_sum_embedding(sum_dim)
        elif self.encode_type == 'concat':
            return self.initialize_concat_embedding(pos_embed_dim, band_embed_dim, prod_embed_dim)
        else:
            raise ValueError(f'encode_type {self.encode_type} is not supported.')


    def initialize_sum_embedding(self, sum_dim) -> T.Tuple[T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor]]:
        # Still need to distinguish between encoder and decoder embeddings, so 
        # will need to pass that value in
        pos_embed_dim = sum_dim
        gsd_patch = {params['GSD']: params['num_patch'] for params in self.input_params.values()}
        gsd_patch = sorted(gsd_patch.items(), key=lambda x: x[0])
        ref_patch_size = gsd_patch[0][-1]
        ref_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(pos_embed_dim, ref_patch_size, cls_token=False)).float()

        # maybe hyperparamter this?
        self.overlap_factor = 1
        # make sure overlap_factor is an odd number
        assert self.overlap_factor % 2 == 1, f'overlap_factor {self.overlap_factor} must be an odd number.'
        pos_embed = {}
        pos_embed[str(gsd_patch[0][0])] = ref_pos_embed
        for gsd, num_patch in gsd_patch[1:]:
            rel_patch_size = int(ref_patch_size//num_patch)
            # This look very long and complicated, it just average over the nearby patch using convolution
            if self.use_superposition_encoding:
                sup_pe_pooler = nn.AvgPool2d(rel_patch_size * self.overlap_factor, stride=rel_patch_size, padding=(self.overlap_factor // 2) * rel_patch_size)
                ref_pos_embed_grid = ref_pos_embed.transpose(1,0).reshape(1,pos_embed_dim,ref_patch_size,ref_patch_size)
                pos_embed[str(gsd)] = sup_pe_pooler(ref_pos_embed_grid).squeeze(0).reshape(pos_embed_dim, -1).transpose(1,0)
            else:
                pos_embed[str(gsd)] = torch.from_numpy(get_2d_sincos_pos_embed(pos_embed_dim, num_patch, cls_token=False)).float()

        band_embed_dim = sum_dim
        if self.embed_band:
            band_embed = nn.Embedding(len(self.groups), band_embed_dim)
        else:
            band_embed = None
        
        prod_embed_dim = sum_dim
        if self.embed_prod:
            unique_prod = {prduct_band.split(':')[0]: None for prduct_band in self.input_params.keys()}
            unique_prod = sorted(unique_prod.keys())
            prod_embed = nn.Embedding(len(unique_prod), prod_embed_dim)
        else:
            prod_embed = None

        pos_embed = nn.ParameterDict(pos_embed).requires_grad_(False)

        return pos_embed, band_embed, prod_embed


    def initialize_concat_embedding(self, pos_embed_dim: int, 
                                          band_embed_dim: int, 
                                          prod_embed_dim: int) -> T.Tuple[T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor], T.Dict[str, torch.Tensor]]:
        # Take dimension as input for MAE implementation where encoder decoder dimension can be different
        # initialize pos_embed
        gsd_patch = {params['GSD']: params['num_patch'] for params in self.input_params.values()}
        gsd_patch = sorted(gsd_patch.items(), key=lambda x: x[0])
        ref_patch_size = gsd_patch[0][-1]
        ref_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(pos_embed_dim, ref_patch_size, cls_token=False)).float()

        # maybe hyperparamter this?
        self.overlap_factor = 1
        # make sure overlap_factor is an odd number
        assert self.overlap_factor % 2 == 1, f'overlap_factor {self.overlap_factor} must be an odd number.'
        pos_embed = {}
        pos_embed[str(gsd_patch[0][0])] = ref_pos_embed
        for gsd, num_patch in gsd_patch[1:]:
            rel_patch_size = int(ref_patch_size//num_patch)
            # This look very long and complicated, it just average over the nearby patch using convolution
            if self.use_superposition_encoding:
                sup_pe_pooler = nn.AvgPool2d(rel_patch_size * self.overlap_factor, stride=rel_patch_size, padding=(self.overlap_factor // 2) * rel_patch_size)
                ref_pos_embed_grid = ref_pos_embed.transpose(1,0).reshape(1,pos_embed_dim,ref_patch_size,ref_patch_size)
                pos_embed[str(gsd)] = sup_pe_pooler(ref_pos_embed_grid).squeeze(0).reshape(pos_embed_dim, -1).transpose(1,0)
            else:
                pos_embed[str(gsd)] = torch.from_numpy(get_2d_sincos_pos_embed(pos_embed_dim, num_patch, cls_token=False)).float()
        
        band_embed = {}
        if self.embed_band:
            #unique_groups = defaultdict(list)
            #for idx, (spectral_group, group_bands) in enumerate(self.groups.items()):
            #    unique_group = ','.join(sorted([band.split(':')[1] for band in group_bands]))
            #    unique_groups[unique_group].append(spectral_group)
            #embed = get_1d_sincos_pos_embed_from_grid(band_embed_dim, torch.arange(len(unique_groups)).numpy())
            #for idx, spectral_groups in enumerate(unique_groups.values()):
            #    for spectral_group in spectral_groups:
            #        if self.pad_band_embed_null:
            #            band_embed[spectral_group] = torch.zeros(embed[idx].shape).float()
            #        else:
            #            band_embed[spectral_group] = torch.from_numpy(embed[idx]).float()
            embed = get_1d_sincos_pos_embed_from_grid(band_embed_dim, torch.arange(len(self.groups)).numpy())
            for i, spectral_group in enumerate(self.groups.keys()):
                band_embed[spectral_group] = torch.from_numpy(embed[i]).float()
        
        prod_embed = {}
        if self.embed_prod:
            unique_prod = {prduct_band.split(':')[0]: None for prduct_band in self.input_params.keys()}
            embed = get_1d_sincos_pos_embed_from_grid(prod_embed_dim, torch.arange(len(unique_prod)).numpy())
            for i, prod in enumerate(unique_prod.keys()):
                if self.pad_prod_embed_null:
                    prod_embed[prod] = torch.zeros_like(torch.from_numpy(embed[i])).float()
                else:
                    prod_embed[prod] = torch.from_numpy(embed[i]).float()
        pos_embed = nn.ParameterDict(pos_embed).requires_grad_(False)
        band_embed = nn.ParameterDict(band_embed).requires_grad_(False)
        prod_embed = nn.ParameterDict(prod_embed).requires_grad_(False)
        
        return pos_embed, band_embed, prod_embed


    def make_patch_grid_concentric_square_selection_index(self, num_patch: int): 
        patch_grid = torch.zeros(num_patch, num_patch)
        n = (num_patch - 1) // 2 +1
        square_side = num_patch
        for i in range(n):
            for j in range(i, num_patch-i):
                squanre_ground_cover = int((square_side / num_patch) * self.ground_cover)
                patch_grid[i,j] = squanre_ground_cover
                patch_grid[num_patch-i-1,j] = squanre_ground_cover
                patch_grid[j,i] = squanre_ground_cover
                patch_grid[j, num_patch-i-1] = squanre_ground_cover
            square_side -= 2
        patch_grid = patch_grid.reshape(-1)
        selection_index = {}
        unique_squanre_ground_cover = torch.unique(patch_grid)
        for squanre_ground_cover in unique_squanre_ground_cover:
            selection_index[int(squanre_ground_cover.item())] = torch.where(patch_grid <= squanre_ground_cover)
        return selection_index


    def initialize_selection_index(self):
        selection_index = {}
        for group_name, group_member in self.groups.items():
            prams = self.input_params[group_member[0]]
            selection_index[group_name] = self.make_patch_grid_concentric_square_selection_index(prams['num_patch'])
        return selection_index


    def initialize_weights(self) -> None:
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def get_auxilliary_embed(self, original_input: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        if self.encode_type == 'sum':
            return self.get_sum_auxilliary_embed(original_input)
        elif self.encode_type == 'concat':
            return self.get_concat_auxilliary_embed(original_input)
        else:
            raise ValueError(f'encode_type {self.encode_type} is not supported.')
    
    
    def get_sum_auxilliary_embed(self, original_input: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        auxilliary_embed = {}
        for group_name, group_member in self.groups.items():
            product_band = group_member[0]
            product, band = product_band.split(':')
            auxilliary_embed[group_name] = self.pos_embed[str(self.input_params[product_band]['GSD'])]
            if self.embed_band:
                group_idx = int(group_name.split('group')[-1])
                auxilliary_embed[group_name] = auxilliary_embed[group_name] + self.band_embed(torch.tensor(group_idx).to(original_input[product_band].device)).unsqueeze(0)
            if self.embed_prod:
                unique_prod = {prduct_band.split(':')[0]: None for prduct_band in self.input_params.keys()}
                unique_prod = sorted(unique_prod.keys())
                product_idx = unique_prod.index(product)
                auxilliary_embed[group_name] = auxilliary_embed[group_name] + self.prod_embed(torch.tensor(product_idx).to(original_input[product_band].device)).unsqueeze(1)

            data = original_input[product_band]
            band_ground_cover = int(data.shape[-1] * self.input_params[product_band]['GSD'])
            if band_ground_cover not in self.selection_index[group_name]:
                raise ValueError(f'Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, ' +
                                 f'which does not fit in a concentric square grid of {self.input_params[product_band]["num_patch"]}x{self.input_params[product_band]["num_patch"]}' +
                                 f'patches for define ground cover {self.ground_cover}x{self.ground_cover}.')
            auxilliary_embed[group_name] = auxilliary_embed[group_name][self.selection_index[group_name][band_ground_cover]]

        return auxilliary_embed


    def get_concat_auxilliary_embed(self, original_input: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        auxilliary_embed = {}
        for group_name, group_member in self.groups.items():
            product_band = group_member[0]
            product, band = product_band.split(':')
            auxilliary_embed[group_name] = self.pos_embed[str(self.input_params[product_band]['GSD'])]
            if self.embed_band:
                auxilliary_embed[group_name] = torch.cat((auxilliary_embed[group_name], self.band_embed[group_name].expand(auxilliary_embed[group_name].shape[0], -1)), dim=-1)
            if self.embed_prod:
                auxilliary_embed[group_name] = torch.cat((auxilliary_embed[group_name], self.prod_embed[product].expand(auxilliary_embed[group_name].shape[0], -1)), dim=-1)
            
            data = original_input[product_band]
            band_ground_cover = int(data.shape[-1] * self.input_params[product_band]['GSD'])
            if band_ground_cover not in self.selection_index[group_name]:
                raise ValueError(f'Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, ' +
                                 f'which does not fit in a concentric square grid of {self.input_params[product_band]["num_patch"]}x{self.input_params[product_band]["num_patch"]}' +
                                 f'patches for define ground cover {self.ground_cover}x{self.ground_cover}.')
            auxilliary_embed[group_name] = auxilliary_embed[group_name][self.selection_index[group_name][band_ground_cover]]

        return auxilliary_embed
    
    
    def aggregate_by_group(self, patch_embed):
        group_embed = {}
        for group_name, group_member in self.groups.items():
            if self.aggr_type == "mean":
                group_embed[group_name] = torch.stack([patch_embed[product_band] for product_band in group_member], -1).mean(-1)
            else:
                group_embed[group_name] = torch.stack([patch_embed[product_band] for product_band in group_member], -1).sum(-1)
        return group_embed


    def forward_encoder(self, x: T.Dict[str, torch.Tensor]) -> torch.Tensor:
        # B: Batch size, H: Height, W: Width
        # D: Embedding dimension, N_n: Number of patches for each product_band
        # T: Sequence length (number of patches * number of spectral bands)
        # x = {'product:band': (B, 1, H, W), ...}
        # extract patches
        patch_embed = self.ind_patch_embed(x)   #{'product:band': (B, N_n, D), ...}

        group_embed = self.aggregate_by_group(patch_embed) # {'group0': (B, N_n, D), ...}

        #group_embed = self.group_patch_embed(x)

        # add additional embedding
        auxilliary_embed = self.get_auxilliary_embed(x) # {'group0': (N_n, D), ...}

        x = torch.cat([group_embed[group_name].add_(auxilliary_embed[group_name]) for group_name in group_embed.keys()], dim=1) # (B, T, D)
        
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, T + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    

    def forward(self, x: T.Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.forward_encoder(x)
        return x

        
class IndPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            ground_cover: int,
            input_params: T.Dict[str, T.Dict[str, int]],
            embed_dim: int = 1024, # Embedding dimension
            norm_layer: nn.Module=None,
            bias: bool=True,
    ) -> None:
        super().__init__()
        proj_dict = {}
        self.input_params = input_params
        self.embed_dim = embed_dim
        # compute data embed size without addtional embeding
        for product_band, params in input_params.items():
            kernel_size = int(ground_cover//float(params['GSD'])//params['num_patch'])
            proj_dict[product_band] = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.patch_embed = nn.ModuleDict(proj_dict)
        self.norm = norm_layer(embed_dim) if norm_layer else None
        
        self.initialize_weights()
        
    
    def initialize_weights(self) -> None:
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for _, conv in self.patch_embed.items():
            w = conv.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


    def forward(self, x: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        # project individual product bands to patch embedding
        patch_embed_dict = {}
        for product_band, data in x.items():
            patch_embed_dict[product_band] = self.patch_embed[product_band](data).flatten(2).transpose(1, 2) 

        if self.norm:
            for product_band, patch_embed in patch_embed_dict.items():
                patch_embed_dict[product_band] = self.norm(patch_embed)
        return patch_embed_dict
    

class GroupPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            ground_cover: int,
            input_params: T.Dict[str, T.Dict[str, int]],
            embed_dim: int = 1024, # Embedding dimension
            norm_layer: nn.Module=None,
            bias: bool=True,
            groups=None,
    ) -> None:
        super().__init__()
        proj_dict = {}
        self.input_params = input_params
        self.embed_dim = embed_dim
        self.groups = groups
        # compute data embed size without addtional embeding
        for group_name, group_member in self.groups.items():
            num_channels = len(group_member)
            params = input_params[group_member[0]]
            kernel_size = int(ground_cover//float(params['GSD'])//params['num_patch'])
            proj_dict[group_name] = nn.Conv2d(num_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.patch_embed = nn.ModuleDict(proj_dict)
        self.norm = norm_layer(embed_dim) if norm_layer else None
        
        self.initialize_weights()
        
    
    def initialize_weights(self) -> None:
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for _, conv in self.patch_embed.items():
            w = conv.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


    def forward(self, x: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        # group product bands together then project to patch embedding
        patch_embed_dict = {}
        for group_name, bands in self.groups.items():
            patch_embed_dict[group_name] = self.patch_embed[group_name](torch.cat([x[band] for band in bands], dim=1)).flatten(2).transpose(1, 2)

        if self.norm:
            for product_band, patch_embed in patch_embed_dict.items():
                patch_embed_dict[product_band] = self.norm(patch_embed)
        return patch_embed_dict


class IndPatchEmbedDeprecated(nn.Module):
    """ 2D independent patch embeding
    this is old design mark for archive
    """
    def __init__(
            self,
            in_chans,
            img_sizes,
            num_patches,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.img_sizes = img_sizes
        self.num_patches = num_patches
        self.flatten = flatten

        self.projs = nn.ModuleList([])
        for in_chan, img_size, num_patch in zip(in_chans, img_sizes, num_patches):
            assert not img_size%num_patch, f'{img_size} cannot evenly split into {num_patch} patches.'
            patch_size = int(img_size//num_patch)
            self.projs.append(nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, groups=in_chan))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        out = []
        for xi, layer in zip(x, self.projs):
            out.append(layer(xi).flatten(2).transpose(1, 2))
        torch.cat(out, dim=1)
        x = self.norm(out)
        return x
    

@MODEL.register_module()
def usat_vit_small_encoder(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=384, depth=12, num_heads=6,
                        embed_band=True, band_embed_dim=32,
                        embed_prod=True, prod_embed_dim=32,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_base_encoder(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=768, depth=12, num_heads=12,
                        embed_band=True, band_embed_dim=64,
                        embed_prod=True, prod_embed_dim=64,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_only_pos_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=False, band_embed_dim=0,
                        embed_prod=False, prod_embed_dim=0,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_only_pos_embed_pad_null(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128, pad_band_embed_null=True,
                        embed_prod=True, prod_embed_dim=128, pad_prod_embed_null=True,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_reweighted(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=252,
                        embed_prod=True, prod_embed_dim=4,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_no_prod_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=256,
                        embed_prod=False, prod_embed_dim=0,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_no_group_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=False, band_embed_dim=0,
                        embed_prod=True, prod_embed_dim=256,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_prod_embed_pad_null(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1024, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128, pad_prod_embed_null=True,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_384(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1152, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=256,
                        embed_prod=True, prod_embed_dim=128,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)


@MODEL.register_module()
def usat_vit_large_encoder_384_no_prod_embed(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1152, depth=24, num_heads=16,
                        embed_band=True, band_embed_dim=384,
                        embed_prod=False, prod_embed_dim=0,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)



@MODEL.register_module()
def usat_vit_huge_encoder(input_parmas, **kwargs):
    model_kwargs = dict(input_params=input_parmas, embed_dim=1280, depth=32, num_heads=16,
                        embed_band=True, band_embed_dim=128,
                        embed_prod=True, prod_embed_dim=128,
                        mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return USatViTEncoder(**model_kwargs)

