from typing import Optional, Iterable, Union, Tuple, List, Mapping, Any
from PIL import Image

import torch
from torch import Tensor
import torchvision as tv

from transformers import AutoTokenizer, AutoConfig, CLIPImageProcessor
from mplug_owl2.model.modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM
from mplug_owl2.mm_utils import process_images

from ..base import MixedPrompts, MultiModalModelBase


class Mplug2(MultiModalModelBase):
    model_path: str = 'MAGAer13/mplug-owl2-llama2-7b'
    template: Tuple[str, ...] = ('USER: ', ' ASSISTANT: ')

    def __init__(self, **hf_kwargs):
        super().__init__(**hf_kwargs)
        self.tokenizer, self._image_preprocessor, self.mm_model = \
            self._init_model()
        self._image_preprocessor.do_normalize = False
        self._image_normalizer = tv.transforms.Normalize(
            mean=self._image_preprocessor.image_mean,
            std=self._image_preprocessor.image_std)
        self.eval().requires_grad_(False)

    def _init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, **self.hf_kwargs)
        config = AutoConfig.from_pretrained(self.model_path, **self.hf_kwargs)
        mm_model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
            self.model_path, config=config, **self.hf_kwargs)
        device, dtype = mm_model.device, mm_model.dtype
        base_model = mm_model.get_model()
        base_model.vision_model.to(device=device, dtype=dtype)
        base_model.visual_abstractor.to(device=device, dtype=dtype)
        preprocessor = CLIPImageProcessor.from_pretrained(
            self.model_path, **self.hf_kwargs)
        self._old_prepare_inputs_labels_for_multimodal = \
            mm_model.prepare_inputs_labels_for_multimodal
        return tokenizer, preprocessor, mm_model

    def image_preprocessor(self, image: Image.Image) -> Tensor:
        return process_images([image], self._image_preprocessor)

    def format_prompt(
        self,
        question: str,
        suffix: Optional[str] = None,
        image: Optional[Tensor] = None,
        response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        p1, p2 = self.template
        prompt: List[Union[str, Tensor]] = [p1]
        if image is not None:
            prompt.append(image)
        prompt.append(question)
        if suffix is not None:
            prompt.append(suffix)
        prompt.append(p2)
        if response is not None:
            prompt.append(response)
        return prompt

    def images_to_embedding(self, images: Tensor) -> Tensor:
        images = self._image_normalizer(images)
        return self.mm_model.encode_images(images)

    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        return self.mm_model.get_model().embed_tokens(ids)

    def _modality_indicator(
        self, mixed_prompts: MixedPrompts,
        toks: Tensor, slices: Iterable[Iterable[slice]],
    ) -> Tensor:
        mis = torch.zeros_like(toks)
        for b, (mp, ms) in enumerate(zip(mixed_prompts, slices)):
            for p, s in zip(mp.sequence, ms):
                if isinstance(p, Tensor):
                    mis[b, s] = 1
        return mis

    def to_embedding(
        self, mixed_prompts: MixedPrompts,
    ) -> Mapping[str, Any]:
        embs = super().to_embedding(mixed_prompts)
        embs['mis'] = self._modality_indicator(
            mixed_prompts, embs['toks'], embs['slices'])
        return embs

    def _embedding_call(
        self, func_name: str, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        def prepare(*args):
            ids, attns, pkvs, labels, _ = args
            if pkvs is not None:
                mis = torch.zeros_like(ids)
                return ids, mis, attns, pkvs, None, labels
            return None, embs['mis'], embs['attns'], None, embs['embs'], labels
        self.mm_model.prepare_inputs_labels_for_multimodal = prepare
        func = getattr(self.mm_model, func_name)
        outputs = func(
            inputs_embeds=embs['embs'], attention_mask=embs['attns'], **kwargs)
        self.mm_model.prepare_inputs_labels_for_multimodal = \
            self._old_prepare_inputs_labels_for_multimodal
        return outputs

    def embedding_forward(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        return self._embedding_call('__call__', embs, **kwargs)

    def embedding_generate(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        kwargs.setdefault('use_cache', True)
        return self._embedding_call('generate', embs, **kwargs)
