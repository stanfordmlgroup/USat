import inspect
from types import FunctionType
from typing import Any, Dict, List, Union, Type, Callable, Optional
import warnings


def default_build_fn (key: str, registry: 'Registry', build_args: Dict):
    if not isinstance(registry, Registry):
        raise TypeError (f'Registry must be a core.Registry object but got {type(registry)}')

    obj = registry.get(key)

    if obj is None:
        raise KeyError(f'Provided key {key} is not available. Registry has: {registry}')
    
    try:
        return obj(**build_args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj.__name__}: {e}')


class RegistryTypeWarning(Warning):
    def __init__(self, message) -> None:
        self.message = message


    def __str__(self) -> str:
        return repr(self.message)


class Registry:
    def __init__(self, name: str, 
                       build_fn: Optional[Callable] = None, 
                       item_types: Optional[List[Type]] = None, 
                       strict: bool = False) -> None:
        self._name = name
        self._container_dict = {}

        if item_types is not None:
            if not isinstance(item_types, list):
                raise TypeError(f'item_types must be a list but got {type(item_types)} instead.')

            for item_type in item_types:
                if not isinstance(item_type, type):
                    raise TypeError(f'type in item_types must be a type but got {type(item_types)} instead.')

        self.item_types = item_types

        if not isinstance(strict, bool):
            raise TypeError(f'strict must be a bool but got {type(item_types)} instead.')
        self.strict = strict

        if build_fn is None:
            self.build_fn = default_build_fn
        else:
            build_fn_args = inspect.signature(build_fn).parameters
            if 'registry' not in build_fn_args:
                raise TypeError(f'build_fn must contain keyword argument "registry".')
            self.build_fn = build_fn


    def __len__(self) -> int:
        return len(self._container_dict)

    
    def __contains__(self, key):
        return self.get(key, default=None) is not None

    
    def __repr__(self) -> str:
        reg_str = ''
        for key, val in self._container_dict.items():
            reg_str += f'\t"{key}": {val}\n'
        mode_str = ''
        if self.item_types is not None:
            if self.strict:
                mode_str = 'Restricted Type:\n'
            else: 
                mode_str = 'Recommended Type:\n'
            for item_type in self.item_types:
                mode_str += f'\t{item_type}\n'

        return (f'REGISTRY({self.name}):\n'
                f'{mode_str}'
                f'Items:\n'
                f'{reg_str}')

    
    def _register_module(self, module_name, module, overwrite):
        if not (isinstance(module, type) or isinstance(module, FunctionType)):
            raise TypeError(f'module must be a class or function but get {type(module)} instead.')
        if module_name is None:
            module_name = module.__name__

        if not overwrite and module_name in self._container_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        
        # check item types
        if self.item_types is not None:
            if isinstance(module, FunctionType):
                if self.strict:
                    raise TypeError(f"Specify types: {self.item_types}, but get function instead.")
                else:
                    warnings.warn(f"Specify types: {self.item_types}, but get function instead.", RegistryTypeWarning)
            else:
                flag = 0
                for item_type in self.item_types:
                    flag += issubclass(module, item_type)
                if flag == 0:
                    if self.strict:
                        raise TypeError(f"Specify types: {self.item_types}, but get {module}")
                    else:
                        warnings.warn(f"Specify types: {self.item_types}, but get {module}", RegistryTypeWarning)

        self._container_dict[module_name] = module
    

    def register_module(self, name: str = None, overwrite: bool = True, module: Union[Type, Callable] = None) -> Union[Type, Callable]:
        # input checking
        if not isinstance(overwrite, bool):
            raise TypeError(f'overwrite must be a boolean, but got {type(overwrite)}')
        
        if not (name is None or isinstance(name, str)):
            raise TypeError('name must be either of None, an instance of str}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(name, module, overwrite)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(name, module, overwrite)
            return module

        return _register

    
    def get(self, key: str, default: Any = None) -> Union[Type, Callable]:
        if key in self._container_dict:
            return self._container_dict[key]
        else:
            return default

    
    def build(self, *args, **kwargs):
        return self.build_fn(*args, **kwargs, registry=self)

    
    @property
    def name (self):
        return self._name
    
    
    @property
    def items (self):
        return self._container_dict

    


        