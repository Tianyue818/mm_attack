from typing import Optional, Iterable, Union, Tuple, Mapping, Any
from PIL import Image

import torch
from torch import Tensor
import torchvision as tv

from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor

from ..base import MultiModalModelBase


class Mplug(MultiModalModelBase):
    model_path: str = 'MAGAer13/mplug-owl-llama-7b'
    template: Tuple[str, ...] = (
        'The following is a conversation between a curious human '
        'and AI assistant. The assistant gives helpful, detailed, '
        'and polite answers to the user\'s questions.\nHuman: ',
        '\nHuman: ',
        '\nAI: ',
    )

    def __init__(self, **hf_kwargs):
        super().__init__(**hf_kwargs)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(
            self.model_path, use_fast=True, **self.hf_kwargs)
        self._image_preprocessor = MplugOwlImageProcessor.from_pretrained(
            self.model_path, **self.hf_kwargs)
        self._image_preprocessor.do_normalize = False
        self._image_normalizer = tv.transforms.Normalize(
            mean=self._image_preprocessor.image_mean,
            std=self._image_preprocessor.image_std)
        self.mm_model = MplugOwlForConditionalGeneration.from_pretrained(
            self.model_path, **self.hf_kwargs)
        self.eval().requires_grad_(False)

    def image_preprocessor(self, image: Image.Image) -> Tensor:
        result = self._image_preprocessor(image)
        return torch.tensor(result.pixel_values[0])

    def format_prompt(
        self,
        question: str,
        suffix: Optional[str] = None,
        image: Optional[Tensor] = None,
        response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        p1, p2, p3 = self.template
        prompt = [p1, question]
        if suffix is not None:
            prompt.append(suffix)
        if image is not None:
            prompt += [p2, image]
        prompt.append(p3)
        if response is not None:
            prompt.append(response)
        return prompt

    def images_to_embedding(self, images: Tensor) -> Tensor:
        images = self._image_normalizer(images)
        embeds = self.mm_model.vision_model(
            images, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(
            embeds.size()[:-1], dtype=torch.long, device=embeds.device)
        bs = embeds.shape[0]
        query_tokens = self.mm_model.query_tokens.expand(bs, -1, -1)
        embeds = self.mm_model.abstractor(
            query_embeds=query_tokens,
            encoder_hidden_states=embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True)['last_hidden_state']
        return embeds

    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        text_embeds = self.mm_model.get_input_embeddings()(ids)
        try:
            transformer = self.mm_model.language_model.transformer
            text_embeds = transformer.word_embeddings_layernorm(text_embeds)
        except AttributeError:
            pass
        return text_embeds

    def embedding_forward(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        return self.mm_model.language_model(
            inputs_embeds=embs['embs'], attention_mask=embs['attns'], **kwargs)

    def embedding_generate(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        kwargs.setdefault('use_cache', True)
        return self.mm_model.language_model.generate(
            inputs_embeds=embs['embs'], attention_mask=embs['attns'], **kwargs)
