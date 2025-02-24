from typing import Optional, Iterable, Tuple, Union

import torch
from torch import Tensor
from PIL import Image
import torchvision as tv
from transformers import (
    AutoTokenizer, AutoProcessor,
    AutoModel, InstructBlipForConditionalGeneration)

from .base import MixedPrompts, MultiModalModelBase


class LavisBase(MultiModalModelBase):
    model_class: type = AutoModel
    model_name: str = NotImplemented
    model_base: str = NotImplemented
    template: Tuple[str, ...] = NotImplemented

    def __init__(self, **hf_kwargs) -> None:
        super().__init__(**hf_kwargs)
        self.model_path = f'Salesforce/{self.model_name}-{self.model_base}'
        processor = AutoProcessor.from_pretrained(self.model_path, **hf_kwargs)
        self.tokenizer = processor.tokenizer
        self._image_preprocessor = processor.image_processor
        self._image_preprocessor.do_normalize = False
        self._image_normalizer = tv.transforms.Normalize(
            mean=self._image_preprocessor.image_mean,
            std=self._image_preprocessor.image_std)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, subfolder="qformer_tokenizer", **hf_kwargs)
        self.mm_model = self.model_class.from_pretrained(
            self.model_path, **hf_kwargs)

    def image_preprocessor(self, image: Image.Image) -> Tensor:
        return self._image_preprocessor(
            image, return_tensors='pt').pixel_values

    def to_embedding(
        self, mixed_prompts: MixedPrompts
    ) -> Tuple[Tensor, Tensor, Tensor, Iterable[Iterable[slice]]]:
        self._queries = tuple(p.mapping['question'] for p in mixed_prompts)
        return super().to_embedding(mixed_prompts)

    def format_prompt(
        self, question: str,
        suffix: Optional[str] = None,
        image: Optional[Tensor] = None,
        response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        if image is None:
            raise ValueError('Image must be provided.')
        t1, t2 = self.template
        prompt = [image, t1, question]
        if suffix is not None:
            prompt.append(suffix)
        prompt.append(t2)
        if response is not None:
            prompt.append(response)
        return prompt

    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        return self.mm_model.get_input_embeddings()(ids)

    def images_to_embedding(self, images: Tensor) -> Tensor:
        images = self._image_normalizer(images)
        image_embs = self.mm_model.vision_model(
            images, return_dict=True).last_hidden_state
        bs, ntoks, _ = image_embs.size()
        image_attns = torch.ones(
            (bs, ntoks), dtype=torch.long, device=image_embs.device)
        qtoks = self.mm_model.query_tokens.expand(bs, -1, -1)
        qattns = torch.ones(
            qtoks.size()[:-1], dtype=torch.long, device=image_embs.device)
        qftoks = self.qformer_tokenizer(
            list(self._queries), padding=True, return_tensors='pt')
        qfattns = torch.cat([qattns, qftoks.attention_mask], dim=1)
        qouts = self.mm_model.qformer(
            input_ids=qftoks.input_ids,
            attention_mask=qfattns,
            query_embeds=qtoks,
            encoder_hidden_states=image_embs,
            encoder_attention_mask=image_attns,
            return_dict=True)
        qouts = qouts.last_hidden_state[:, : qtoks.size(1), :]
        return self.mm_model.language_projection(qouts)

    def embedding_forward(
        self, embs: Tensor, attns: Tensor, **kwargs
    ) -> Tensor:
        return self.mm_model.language_model(
            inputs_embeds=embs, attention_mask=attns, **kwargs)

    def embedding_generate(
        self, embs: Tensor, attns: Tensor, **kwargs
    ) -> Tensor:
        outputs = self.mm_model.language_model.generate(
            inputs_embeds=embs, attention_mask=attns, **kwargs)
        arch = self.mm_model.config.text_config.architectures[0]
        if arch == "LLaMAForCausalLM":
            if isinstance(outputs, torch.Tensor):
                outputs[outputs == 0] = 2
            else:
                outputs.sequences[outputs.sequences == 0] = 2
        return outputs


class Blip2Base(LavisBase):
    model_name: str = 'blip2'
    template: Tuple[str, ...] = ('', '')


class Blip2FlanT5XL(Blip2Base):
    model_base: str = 'flan-t5-xl'


class Blip2FlanT5XXL(Blip2Base):
    model_base: str = 'flan-t5-xxl'


class Blip2Opt3b(Blip2Base):
    model_base: str = 'opt-2.7b'


class Blip2Opt7b(Blip2Base):
    model_base: str = 'opt-6.7b'


class Blip2Opt7bCoco(Blip2Base):
    model_base: str = 'opt-6.7b-coco'


class InstructBlipBase(LavisBase):
    model_class: type = InstructBlipForConditionalGeneration
    model_name: str = 'instructblip'
    template: Tuple[str, ...] = ('Question: ', ' Answer: ')

    def embedding_forward(
        self, embs: Tensor, attns: Tensor, **kwargs
    ) -> Tensor:
        return self.mm_model.language_model(
            inputs_embeds=embs, attention_mask=attns,
            decoder_inputs_embeds=embs, **kwargs)


class InstructBlipFlanT5XL(InstructBlipBase):
    model_base: str = 'flan-t5-xl'


class InstructBlipFlanT5XXL(InstructBlipBase):
    model_base: str = 'flan-t5-xxl'


class InstructBlipVicuna7b(InstructBlipBase):
    model_base: str = 'vicuna-7b'


class InstructBlipVicuna13b(InstructBlipBase):
    model_base: str = 'vicuna-13b'
