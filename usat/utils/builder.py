import typing as T
import re

import torch
import torchvision.transforms as transforms

from usat.core.registry import Registry
from usat.utils.helper import extract_model_state_dict_from_ckpt


class build_target_from_cfg:
    def __init__(self, target) -> None:
        # TODO: build custom build function for model and pretrian_transform
        self.target = target
    

    def __call__(self, cfg: T.Dict[str, T.Any], registry: 'Registry') -> T.Any:
        build_cfg = cfg.get(self.target, None)
        
        if build_cfg is None:
            raise KeyError(f"Provided cfg does not contain {self.target} config.")
        
        build_cfg = build_cfg.copy()
        key = build_cfg.pop("name")
        obj = registry.get(key)
        
        if obj is None:
            raise KeyError(f'Provided key {key} is not available. Registry has: ', registry)

        return obj(**build_cfg)

    
def task_build_fn(cfg: T.Dict[str, T.Any], registry: 'Registry') -> T.Any:
    task_name = cfg.get('task')
    task_class = registry.get(task_name)

    if task_class is None:
        raise KeyError(f'{task_class} is not available. Registry has: ', registry)
    
    return task_class(cfg)


def model_build_fn(cfg: T.Dict[str, T.Any], registry: 'Registry') -> T.Any:
    model_list = cfg.get('models')

    if model_list is None:
        raise KeyError(f"Provided cfg does not contain model config.")

    models = {}
    for model_name, model_cfg in model_list.items():
        model_type = model_cfg.get('type')
        model_args = model_cfg.get('args')

        model_class = registry.get(model_type)
        if model_class is None:
            raise KeyError(f'{model_class} is not available. Registry has: ', registry)

        ckpt = model_cfg.get('ckpt', None)
        ckpt_ignore = model_cfg.get('ckpt_ignore', [])
        ckpt_copy = model_cfg.get('ckpt_copy', [])
        ckpt_remap = model_cfg.get('ckpt_remap', {})
        strict = model_cfg.get('strict', True)
        target_model = model_cfg.get('target_model', model_name)

        model =  model_class(**model_args)

        if ckpt is not None:
            # NOTE: downside, it will try to load ckpt every single time. 
            # but support loadingt different from ckpt for different modules
            ckpt = torch.load(ckpt, map_location='cpu')
            model_state_dict = extract_model_state_dict_from_ckpt(ckpt)[target_model]

            # Pop all of the keys that can be ignored
            new_keys = list(model_state_dict.keys())
            for rgx_item in ckpt_ignore:
                re_expr = re.compile(rgx_item)
                new_keys = [key for key in new_keys if not re_expr.match(key)]
            model_state_dict = dict((k, model_state_dict[k]) for k in new_keys)

            # Add in all keys you want to copy the default param for
            for copy_key in ckpt_copy:
                print(f'Skipping model load for: {copy_key}')
                model_state_dict[copy_key] = model.state_dict()[copy_key]

            # Remap certain keys for multi sensor pretrain to single sensor finetune
            for key, cfg_map in ckpt_remap.items():
                print(f'Remapping key for custom load: {key}')
                old_val = model_state_dict.pop(key)
                new_val = old_val
                # Apply modifications
                new_name = cfg_map.get("name", key)
                params = cfg_map.get("params", {})
                func = cfg_map.get("func", None)
                if func == "index_select":
                    new_val = torch.index_select(old_val, params['dim'], torch.tensor(params['indices']))
                elif func == "concat_passthrough":
                    new_val = torch.cat((new_val, torch.index_select(model.state_dict()[new_name], params['dim'], torch.tensor(params['index']))), dim=params['dim'])
                    
                model_state_dict[new_name] = new_val
            

            model.load_state_dict(model_state_dict, strict=strict)
            print(f'Custom weight loaded for {model_name}')
        
        models[model_name] = model
    
    return models


def dataset_build_fn(cfg: T.Dict[str, T.Any], registry: 'Registry', split: str = 'train') -> T.Any:
    dataset_cfg = cfg.get('dataset')
    if dataset_cfg is None:
        raise KeyError(f"Provided cfg does not contain datset config.")

    build_cfg = dataset_cfg.copy()
    key = build_cfg.pop("name")
    dataset_class = registry.get(key)

    if dataset_class is None:
        raise KeyError(f'Provided key {key} is not available. Registry has: ', registry)

    dataset_args = build_cfg.get(f'{split}_args', {})

    if dataset_args.get('custom_transform', None):
        transform_args = {'pretrain_transform':dataset_args}
        custom_trans = PRETRAIN_TRANSFORM.build(transform_args, target='custom_transform')
    else:
        custom_trans = None
    
    dataset_args['custom_transform'] = custom_trans

    return dataset_class(**dataset_args)


def compose_transform_build_fn(cfg: T.Dict[str, T.Any], registry: 'Registry', target: str = 'transform1') -> T.Any:
    transform_cfg = cfg.get('pretrain_transform', None)
    if transform_cfg is None:
        raise KeyError(f"Provided cfg does not contain transform config.")
    
    target_cfg = transform_cfg.get(target, None)
    if target_cfg is None: 
        raise KeyError(f"Provided cfg does not contain {target} config.")
    
    transform_list  = []
    for transform_name, transform_args in target_cfg.items():
        transform_class = registry.get(transform_name)

        if transform_class is None:
            raise KeyError(f'Provided key {transform_name} is not available. Registry has: ', registry)
        random_apply_p = transform_args.pop('random_apply', None)
        if random_apply_p is None:
            transform_list.append(transform_class(**transform_args))
        else:
            transform_list.append(transforms.RandomApply([transform_class(**transform_args)], p=random_apply_p))


    return transforms.Compose(transform_list)


DATASET = Registry('dataset', build_fn=dataset_build_fn)
MODEL = Registry('model', build_fn=model_build_fn)
TASK = Registry('task', build_fn=task_build_fn)
PRETRAIN_TRANSFORM = Registry('pretrain_transform', build_fn=compose_transform_build_fn)