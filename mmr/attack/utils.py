import functools
from datetime import datetime
from typing import (
    Optional, Union, Literal,
    Sequence, Iterable, Tuple, List, Dict, SupportsFloat)

import torch
from torch import Tensor

from ..utils import T, Device
from ..models import MixedPrompts


ThreatModel = Literal['linf', 'l2']

DATETIME_FORMAT = '%Y-%m-%d-%H-%M-%S'


def datetime2str(dt: datetime) -> str:
    return dt.strftime(DATETIME_FORMAT)


def str2datetime(s: str) -> datetime:
    return datetime.strptime(s, DATETIME_FORMAT)


def tslice(
    ts: Iterable[Optional[Tensor]], indices: Tensor
) -> Tuple[Optional[Tensor], ...]:
    return tuple(t[indices] if t is not None else None for t in ts)


def untuple(values: Iterable[T]) -> Union[T, Tuple[T, ...]]:
    values = tuple(values)
    if len(values) == 1:
        return values[0]
    return values


def detach(
    *tensors: Optional[Tensor], device: Optional[Device] = None
) -> Union[Optional[Tensor], Tuple[Optional[Tensor], ...]]:
    if device is None:
        return untuple(t.detach() if t is not None else None for t in tensors)
    return untuple(
        t.detach().to(device) if t is not None else None for t in tensors)


def clip(image: Tensor, pert: Tensor, norm: ThreatModel, eps: float) -> Tensor:
    if norm == 'linf':
        pert = torch.clamp(pert, -eps, eps)
    elif norm == 'l2':
        pert = eps * pert / torch.norm(pert)
    else:
        raise NotImplementedError(f'Unknown norm {norm}.')
    return torch.clamp(image + pert, 0.0, 1.0)


def batch_slice(
    inputs: Tensor, slices: Iterable[slice], offset: int = 0
) -> Tensor:
    outputs = []
    for i, s in zip(inputs, slices):
        start, stop = s.start, s.stop
        outputs.append(i[start + offset:stop + offset])
    return torch.cat(outputs, dim=0)


def image_slices(
    prompts: MixedPrompts, slices: Iterable[Sequence[slice]]
) -> List[slice]:
    islices: List[slice] = []
    for prompt, prompt_slices in zip(prompts, slices):
        for p, s in zip(prompt, prompt_slices):
            if isinstance(p, Tensor):
                islices.append(s)
                break
    return islices


def text_slices(
    prompts: MixedPrompts, slices: Iterable[Iterable[slice]],
    texts: Iterable[str]
) -> List[slice]:
    tslices: List[slice] = []
    for prompt, prompt_slices, t in zip(prompts, slices, texts):
        for p, s in zip(prompt, prompt_slices):
            if p == t:
                tslices.append(s)
                break
        else:
            raise ValueError(
                f'Could not find text {t!r} in prompt {prompt!r}.')
    return tslices


def update_embedding_slices(
    embs: Tensor,
    slice_or_slices: Union[slice, Iterable[slice]],
    values: Tensor,
    inplace: bool = False
) -> Tensor:
    """
    This function finds and replaces the embeddings of the specified slices
    in the batched embedding tensor `embs` with the given `value`.

    Args:
        embs: A tensor of shape (batch_size, num_tokens, ...).
        slice_or_slices:
            A slice, or a list of slices for each element in batch.
        value: A tensor of shape (batch_size, max_value_tokens, embedding_dim).

    Returns:
        A tensor of shape (batch_size, num_tokens, embedding_dim).

    Example:
        >>> embs = torch.arange(30).view(2, 5, 3)
        >>> slices = [slice(0, 3), slice(1, 2)]
        >>> value = torch.arange(18).view(2, 3, 3)
        >>> update_embedding(embs, slices, value)
        tensor([[[  0,  -1,  -2],
                [ -3,  -4,  -5],
                [ -6,  -7,  -8]],
        <BLANKLINE>
                [[ -9, -10, -11],
                [-12, -13, -14],
                [-15, -16, -17]]])
    """
    if not inplace:
        embs = embs.detach().clone()
    ss = slice_or_slices
    if isinstance(ss, slice):
        # single slice
        embs[:, ss] = values
    elif len(set(s.start for s in ss)) == len(set(s.stop for s in ss)) == 1:
        # all slices are the same
        embs[:, list(ss)[0]] = values
    else:
        # different slices
        for b, (s, v) in enumerate(zip(ss, values)):
            embs[b, s] = v[:s.stop - s.start]
    return embs


def levenshtein_distance(x: Iterable[T], y: Iterable[T]) -> int:
    x, y = list(x), list(y)
    if len(x) < len(y):
        return levenshtein_distance(y, x)
    if len(y) == 0:
        return len(x)
    previous_row = range(len(y) + 1)
    for i, c1 in enumerate(x):
        current_row = [i + 1]
        for j, c2 in enumerate(y):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def average_levenshtein_distance(
    outputs: Iterable[Iterable[T]], targets: Iterable[Iterable[T]]
) -> float:
    dist = count = 0
    for o, t in zip(outputs, targets):
        o, t = list(o), list(t)
        l = min(len(o), len(t))
        if l == 0:
            dist = 1.0
        else:
            dist += levenshtein_distance(o[:l], t[:l]) / l
        count += 1
    return dist / count


def topk_count(
    output: Tensor, label: Tensor, k: Iterable[int] = (1, )
) -> Tuple[int, ...]:
    k = list(k)
    pred = output.topk(max(k), 1, True, True).indices.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    return tuple(int(correct[:tk].sum()) for tk in k)


@functools.total_ordering
class AccuracyCounter:
    k: List[int]
    corrects: Dict[int, int]
    size: int

    def __init__(self, k: Iterable[int] = (1, )) -> None:
        super().__init__()
        self.k = list(k)
        self.reset()

    def reset(self) -> None:
        self.corrects = {i: 0 for i in self.k}
        self.size = 0

    def add(self, output: Tensor, label: Tensor) -> None:
        self.size += label.shape[0]
        for i, a in zip(self.k, topk_count(output, label, self.k)):
            self.corrects[i] += a

    @property
    def accuracies(self) -> Dict[int, float]:
        if self.size == 0:
            return {k: float('nan') for k in self.k}
            # raise ValueError('No accumulated statistics.')
        return {k: c / self.size for k, c in self.corrects.items()}

    @property
    def errors(self) -> Dict[int, float]:
        return {k: 1 - a for k, a in self.accuracies.items()}

    def __float__(self) -> float:
        return float(self.accuracies[0])

    def __eq__(self, other: SupportsFloat):
        return float(self) == float(other)

    def __lt__(self, other: SupportsFloat):
        return float(self) < float(other)

    def __gt__(self, other: SupportsFloat):
        return float(self) > float(other)

    def __format__(self, mode: Optional[str] = '.2%') -> str:
        return ', '.join(
            f'top{k}={v:{mode}}' for k, v in zip(self.k, self.accuracies))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self})'
