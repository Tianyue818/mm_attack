from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Union, Optional, Tuple, List, Dict, Iterable, Callable, Mapping, Any)

import torch
from torch import Tensor
import torchvision as tv
from PIL import Image
from transformers import (
    PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding)

from ..utils import Device
from .utils import padded_tensor, disable_random_init


@dataclass
class MixedPrompt:
    sequence: List[Union[str, Tensor]]
    mapping: Dict[str, Union[str, Tensor]]

    def __getitem__(self, index: int) -> Union[str, Tensor]:
        return self.sequence[index]


MixedPrompts = Iterable[MixedPrompt]


class MultiModalModelBase(ABC, torch.nn.Module):
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    image_size: int
    _image_transforms: Callable[[Image.Image], Tensor]

    def __init__(self, **hf_kwargs) -> None:
        super().__init__()
        self.hf_kwargs = hf_kwargs

    def requires_grad_(self, requires_grad: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def to_default_dtype(
        self, model: Optional[torch.nn.Module] = None
    ) -> torch.nn.Module:
        model = model or self
        if (dtype := self.hf_kwargs.get('torch_dtype')) is not None:
            model.to(dtype)
        return model

    def raw_image_to_tensor(
        self, image: Union[str, Image.Image], device: Optional[Device] = None
    ) -> Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        tensor = self.image_preprocessor(image)
        if tensor.ndim == 3:
            pass
        elif tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        else:
            raise ValueError(
                'Expected image tensor to have 3 or 4 dimensions, '
                f'got {tensor.ndim}.')
        if device is not None:
            tensor = tensor.to(device)
        if (dtype := self.hf_kwargs.get('torch_dtype')) is not None:
            tensor = tensor.to(dtype)
        return tensor

    def image_preprocessor(self, image: Image.Image) -> Tensor:
        try:
            return self._image_transforms(image)
        except AttributeError:
            pass
        self._image_transforms = tv.transforms.Compose([
            tv.transforms.Resize(
                (self.image_size, ) * 2,
                interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.ToTensor(),
        ])
        return self._image_transforms(image)

    def tokenize(self, text: Iterable[str]) -> BatchEncoding:
        text = list(text)
        return self.tokenizer(
            text, return_tensors='pt', padding=True, add_special_tokens=False)

    def untokenize(
        self, tokens: Tensor, skip_special_tokens: bool = False
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            tokens, skip_special_tokens=skip_special_tokens)

    def format_prompts(
        self, questions: Iterable[str],
        suffixes: Optional[Iterable[Optional[str]]] = None,
        images: Optional[
            Union[Tensor, Iterable[Optional[Tensor]]]
        ] = None,
        responses: Optional[Iterable[Optional[str]]] = None,
    ) -> MixedPrompts:
        questions = list(questions)
        bs = len(questions)
        if responses is not None:
            responses = list(responses)
        else:
            responses = [None] * bs
        if suffixes is not None:
            suffixes = list(suffixes)
        else:
            suffixes = [None] * bs
        if images is not None:
            if isinstance(images, Tensor) and images.ndim == 3:
                raise ValueError(
                    f'Expected images to have 4 dimensions, got {images.ndim}.')
            images = list(images)
        else:
            images = [None] * bs
        if not (len(questions) == len(suffixes) == len(images) == len(responses)):
            raise ValueError(
                'Expected questions, suffxies, images, and responses '
                'to have the same length, '
                f'got {len(questions)}, {len(images)}, {len(suffixes)} '
                f'and {len(responses)} respectively.')
        prompts = []
        for q, s, i, r in zip(questions, suffixes, images, responses):
            seq = list(self.format_prompt(q, s, i, r))
            m = {'question': q, 'suffix': s, 'image': i, 'response': r}
            prompts.append(MixedPrompt(seq, m))
        return prompts

    def _to_mixed_attrs(
        self, mixed_prompts: MixedPrompts
    ) -> Tuple[List[List[Tuple[str, int]]], List[str], List[Tensor]]:
        text_prompts = []
        image_prompts = []
        mixed_attrs = []
        for mixed_prompt in mixed_prompts:
            ma = []
            for prompt in mixed_prompt:
                if isinstance(prompt, str):
                    ma.append(('text', len(text_prompts)))
                    text_prompts.append(prompt)
                elif isinstance(prompt, Tensor):
                    ma.append(('image', len(image_prompts)))
                    if prompt.ndim == 3:
                        prompt = prompt.unsqueeze(0)
                    elif prompt.ndim != 4:
                        raise ValueError(
                            'Expected image prompt to have 3 or 4 dimensions, '
                            f'got {prompt.ndim}.')
                    image_prompts.append(prompt)
                else:
                    raise TypeError(
                        f'Unsupported type for prompt {prompt!r}.')
            mixed_attrs.append(ma)
        return mixed_attrs, text_prompts, image_prompts

    def _to_embedding_by_type(
        self, texts: Iterable[str], images: Optional[Iterable[Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        text_enc = self.tokenize(texts)
        input_ids = text_enc.input_ids
        if input_ids.ndim != 2:
            raise ValueError(
                'Expected input_ids to have 2 dimensions, '
                f'got {input_ids.ndim}.')
        text_embs = self.tokens_to_embedding(input_ids)
        text_attns = text_enc.attention_mask
        if images is not None:
            images = torch.cat(list(images), dim=0)
            if images.ndim != 4:
                raise ValueError(
                    f'Expected images to have 4 dimensions, got {images.ndim}.')
            image_embs = self.images_to_embedding(images)
            image_attns = torch.ones_like(image_embs[..., 0])
        else:
            image_embs = image_attns = torch.empty(0)
        return input_ids, text_embs, text_attns, image_embs, image_attns

    def _to_padded_mixed_embs(
        self, mixed_attrs: List[List[Tuple[str, int]]],
        input_ids: Tensor, text_embs: Tensor, text_attns: Tensor,
        image_embs: Tensor, image_attns: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, List[List[slice]]]:
        mixed_embs, mixed_toks, mixed_slices = [], [], []
        for mix in mixed_attrs:
            toks, embs, pos, slices = [], [], 0, []
            for t, i in mix:
                if t == 'text':
                    att = text_attns[i].bool()
                    e = text_embs[i][att]
                    ids = input_ids[i][att]
                else:
                    att = image_attns[i].bool()
                    e = image_embs[i][att]
                    ids = torch.full_like(att.int(), -1)[att]
                toks.append(ids)
                embs.append(e)
                slices.append(slice(pos, pos := pos + e.shape[0]))
            mixed_toks.append(torch.cat(toks, dim=0))
            mixed_embs.append(torch.cat(embs, dim=0))
            mixed_slices.append(slices)
        padded_toks = padded_tensor(mixed_toks, -2)
        padded_embs = padded_tensor(mixed_embs, 0)
        attention_masks = (padded_toks != -2).int()
        return padded_embs, padded_toks, attention_masks, mixed_slices

    def to_embedding(
        self, mixed_prompts: MixedPrompts
    ) -> Dict[str, Union[Tensor, Iterable[Iterable[slice]]]]:
        mixed_attrs, texts, images = \
            self._to_mixed_attrs(mixed_prompts)
        input_ids, text_embs, text_attns, image_embs, image_attns = \
            self._to_embedding_by_type(texts, images)
        padded_embs, padded_toks, attention_masks, mixed_slices = \
            self._to_padded_mixed_embs(
                mixed_attrs, input_ids, text_embs, text_attns,
                image_embs, image_attns)
        return {
            'embs': padded_embs,
            'toks': padded_toks,
            'attns': attention_masks,
            'slices': mixed_slices,
        }

    def forward(self, mixed_prompts: MixedPrompts, **kwargs) -> Tensor:
        embs = self.to_embedding(mixed_prompts)
        return self.embedding_forward(embs, **kwargs)

    def generate(
        self, mixed_prompts: MixedPrompts,
        max_new_tokens: int = 300, num_beams: int = 1,
        min_length: int = 1, top_p: float = 0.9,
        repetition_penalty: float = 1.0, length_penalty: float = 1.0,
        temperature: float = 1.0,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        with torch.no_grad():
            embs = self.to_embedding(mixed_prompts)
            outputs = self.embedding_generate(
                embs,
                max_new_tokens=max_new_tokens, stopping_criteria=None,
                num_beams=num_beams, do_sample=True, min_length=min_length,
                top_p=top_p, repetition_penalty=repetition_penalty,
                length_penalty=length_penalty, temperature=temperature,
                **kwargs)
        return self.untokenize(
            outputs, skip_special_tokens=skip_special_tokens)

    @abstractmethod
    def format_prompt(
        self, question: str, suffix: Optional[str] = None,
        image: Optional[Tensor] = None, response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def images_to_embedding(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def embedding_forward(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def embedding_generate(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        raise NotImplementedError
