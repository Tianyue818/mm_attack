import contextlib
from typing import Iterable, Any, Mapping, Generator

import torch
from torch import Tensor

from ..utils import cached_file


def cached_load(
    model_url: str, check_hash: bool = True, progress: bool = True,
    map_location: str = 'cpu', weights_only: bool = False,
) -> Mapping[str, Any]:
    """Load a model from cache if possible, otherwise download it.  """
    path = cached_file(model_url, check_hash, progress)
    return torch.load(
        path, map_location=map_location, weights_only=weights_only)


def padded_tensor(tensor: Iterable[Tensor], pad_value: Any) -> Tensor:
    return torch.nested.to_padded_tensor(
        torch.nested.as_nested_tensor(list(tensor)), pad_value)


DISABLE_RANDOM_INIT_NAMES = (
    'uniform_', 'normal_', 'xavier_uniform_', 'xavier_normal_',
    'kaiming_uniform_', 'kaiming_normal_', 'sparse_', 'uniform', 'normal',
    'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
)


_random_init_funcs = {}


def disable_random_init(disable=DISABLE_RANDOM_INIT_NAMES) -> None:
    do_nothing = lambda *args, **kwargs: None
    for name in dir(torch.nn.init):
        if name not in disable:
            continue
        if name in _random_init_funcs:
            continue
        func = getattr(torch.nn.init, name)
        if not callable(func):
            continue
        _random_init_funcs[name] = func
        setattr(torch.nn.init, name, do_nothing)


def enable_random_init(disable=DISABLE_RANDOM_INIT_NAMES) -> None:
    for name in list(_random_init_funcs):
        if name not in disable:
            continue
        setattr(torch.nn.init, name, _random_init_funcs.pop(name))


@contextlib.contextmanager
def disable_random_init_context(
    disable=DISABLE_RANDOM_INIT_NAMES
) -> Generator[None, Any, None]:
    disable_random_init(disable)
    yield
    enable_random_init(disable)
