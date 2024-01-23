import json
from types import SimpleNamespace
from typing import Dict, Any, List, Union

import yaml
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    # wrapper to covert np to json
    def default(self, obj):
        if (
            isinstance(obj, np.ndarray)
            or isinstance(obj, np.floating)
            or isinstance(obj, np.number)
        ):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_json(path: str) -> Union[Dict[str, Any], List]:
    with open(path, "r") as f:
        o = json.load(f)
    return o


def write_json(o: Union[Dict[str, Any], List], path: str) -> None:
    with open(path, "w") as f:
        json.dump(o, f, cls=NumpyEncoder)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Dict[str, Any], path: str) -> str:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)
    return path


def namespace_to_dict(ns: SimpleNamespace) -> Dict[str, Any]:
    c = ns.__dict__
    for k, v in c.items():
        if isinstance(v, SimpleNamespace):
            c[k] = namespace_to_dict(v)
    return c


def dict_to_namespace(c: Dict[str, Any]) -> SimpleNamespace:
    for k, v in c.items():
        if isinstance(v, dict):
            c[k] = dict_to_namespace(v)
    ns = SimpleNamespace(**c)
    return ns
