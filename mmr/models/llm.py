from typing import Optional, Tuple, Type, List, Mapping, Any, Iterable, Union

from torch import Tensor
from transformers import (
    PreTrainedTokenizer, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM,
    LlamaTokenizer, LlamaForCausalLM)

from .base import MultiModalModelBase


class CausalLM(MultiModalModelBase):
    tokenizer_cls: Type[PreTrainedTokenizer] = AutoTokenizer
    tokenizer: PreTrainedTokenizer
    language_model_cls: Type[PreTrainedModel] = AutoModelForCausalLM
    language_model: PreTrainedModel
    language_model_path: str = NotImplemented
    template: List[str] = NotImplemented

    def __init__(
        self,
        model: Optional[str] = None,
        template: Optional[List[str]] = None,
        **hf_kwargs: Mapping[str, Any]
    ) -> None:
        super().__init__(**hf_kwargs)
        self.template = template or self.template
        self.language_model_path = model or self.language_model_path
        self.tokenizer, self.language_model = self._init_language_model()
        self.eval().requires_grad_(False)

    def _init_language_model(
        self
    ) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        tokenizer = self.tokenizer_cls.from_pretrained(
            self.language_model_path, use_fast=True, **self.hf_kwargs)
        tokenizer.pad_token = '<unk>'
        tokenizer.padding_side = 'left'
        model = self.language_model_cls.from_pretrained(
            self.language_model_path, **self.hf_kwargs)
        return tokenizer, model

    def format_prompt(
        self,
        question: str,
        suffix: Optional[str] = None,
        image: Optional[Tensor] = None,
        response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        if image is not None:
            raise NotImplementedError(
                f'{self.language_model_path} does not support image inputs.')
        suffix = '' if suffix is None else suffix
        response = '' if response is None else response
        return [
            t.format(question=question, suffix=suffix, response=response)
            for t in self.template]

    def images_to_embedding(self, image: Tensor) -> Tensor:
        raise NotImplementedError(
            f'{self.language_model_path} does not support image inputs.')

    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        return self.language_model.base_model.embed_tokens(ids)

    def embedding_forward(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        return self.language_model(
            inputs_embeds=embs['embs'], attention_mask=embs['attns'], **kwargs)

    def embedding_generate(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        kwargs.setdefault('use_cache', True)
        return self.language_model.generate(
            inputs_embeds=embs['embs'], attention_mask=embs['attns'], **kwargs)


class Llama2(CausalLM):
    tokenizer_cls = LlamaTokenizer
    language_model_cls = LlamaForCausalLM
    language_model_path = 'meta-llama/Llama-2-7b-chat-hf'
    template = ['<s>[INST] {question}', '{suffix}', ' [/INST] {response}']
