import re
from typing import Optional, Tuple, Union, Iterable, Mapping, Any

from PIL import Image
import torch
from torch import Tensor
import torchvision as tv
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    PreTrainedModel, CLIPImageProcessor)
from peft import PeftModel

from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, get_model_name_from_path

from .base import MultiModalModelBase, MixedPrompt
from ..utils import hf_cached_root


class LLaVABase(MultiModalModelBase):
    model_path: str = NotImplemented
    model_base: str = NotImplemented
    _image_config = {'image_aspect_ratio': 'pad'}
    template = [
        'A chat between a curious human '
        'and an artificial intelligence assistant. '
        'The assistant gives helpful, detailed, '
        "and polite answers to the human's questions. USER: ",
        '\n',
        ' ASSISTANT:'
    ]

    def __init__(self, **hf_kwargs):
        super().__init__(**hf_kwargs)
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self._image_preprocessor, self.mm_model = \
            self._init_model()
        self._image_preprocessor.do_normalize = False
        self._image_normalizer = tv.transforms.Normalize(
            mean=self._image_preprocessor.image_mean,
            std=self._image_preprocessor.image_std)

    def _init_model(self) -> Tuple[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        CLIPImageProcessor, PreTrainedModel
    ]:
        model_base_path = hf_cached_root(self.model_base)
        tokenizer = AutoTokenizer.from_pretrained(
            model_base_path, use_fast=True, **self.hf_kwargs)
        model_path = hf_cached_root(self.model_path)
        config = AutoConfig.from_pretrained(model_path, **self.hf_kwargs)
        config.mm_vision_tower = hf_cached_root(config.mm_vision_tower)
        mm_model = LlavaLlamaForCausalLM.from_pretrained(
            self.model_base, config=config, **self.hf_kwargs)
        device, dtype = mm_model.device, mm_model.dtype
        # projector
        lm_head = mm_model.lm_head
        token_num = mm_model.lm_head.out_features
        tokem_dim = mm_model.lm_head.in_features
        if self.lora:
            # this stupid condition shouldn't be possible
            # code adapted from llava.model.builder.load_pretrained_model
            if lm_head.weight.shape[0] != token_num:
                weight = torch.empty(
                    token_num, tokem_dim, device=device, dtype=dtype)
                lm_head.weight = torch.nn.Parameter(weight)
                mm_model.model.embed_tokens.weight = \
                    torch.nn.Parameter(torch.empty_like(weight))
            cache_file = hf_hub_download(
                self.model_path, 'non_lora_trainables.bin',
                local_files_only=self.hf_kwargs.get('local_files_only', False))
            state = torch.load(cache_file, map_location='cpu')
            state = {
                re.sub(r'^(base_model\.|model\.model\.)+', 'model.', k): v
                for k, v in state.items()}
            mm_model.load_state_dict(state, strict=False)
            mm_model = PeftModel.from_pretrained(
                mm_model, model_path, **self.hf_kwargs)
            mm_model = mm_model.merge_and_unload()
        else:
            # proj_state = torch.load(
            #     os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            # model.load_state_dict(proj_state, strict=False)
            raise NotImplementedError(
                "I am confused about LLaVA's projector weight location.")
        vision_tower = mm_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=dtype)
        image_preprocessor = vision_tower.image_processor
        self._old_prepare_inputs_labels_for_multimodal = \
            mm_model.prepare_inputs_labels_for_multimodal
        return tokenizer, image_preprocessor, mm_model

    def image_preprocessor(self, image: Image.Image) -> Tensor:
        return process_images(
            image, self._image_preprocessor, self._image_config)

    def format_prompt(
        self,
        question: str,
        suffix: Optional[str] = None,
        image: Optional[Tensor] = None,
        response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        p1, p2, p3 = self.template
        prompt = [p1]
        if image is not None:
            prompt += [image, p2]
        prompt.append(question)
        if suffix is not None:
            prompt.append(suffix)
        prompt.append(p3)
        if response is not None:
            prompt.append(response)
        return prompt

    def images_to_embedding(self, images: Tensor) -> Tensor:
        images = self._image_normalizer(images)
        vision_tower = self.mm_model.get_vision_tower()
        # llava wrapps vision_tower.forward with a torch.no_grad() decorator,
        # removing it for gradient computation
        features = vision_tower.forward.__wrapped__(vision_tower, images)
        return self.mm_model.get_model().mm_projector(features)

    def tokens_to_embedding(self, ids: Tensor) -> Tensor:
        return self.mm_model.get_model().embed_tokens(ids)

    def _embedding_call(
        self, func_name: str, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        def prepare(*args):
            ids, attns, pkvs, labels, _ = args
            if pkvs is not None:
                return ids, attns, pkvs, None, labels
            return None, embs['attns'], None, embs['embs'], labels
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
        return self._embedding_call('forward', embs, **kwargs)

    def embedding_generate(
        self, embs: Mapping[str, Any], **kwargs
    ) -> Tensor:
        kwargs.setdefault('use_cache', True)
        return self._embedding_call('generate', embs, **kwargs)


class LLaVA15_7b(LLaVABase):
    size: int = 7
    lora: bool = False

    @property
    def model_path(self) -> str:
        lora = '-lora' if self.lora else ''
        return f'liuhaotian/llava-v1.5-{self.size}b{lora}'

    @property
    def model_base(self) -> str:
        return f'lmsys/vicuna-{self.size}b-v1.5'


class LLaVA15_7b_LoRA(LLaVA15_7b):
    lora = True


class LLaVA15_13b_LoRA(LLaVA15_7b):
    size = 13
    lora = True


class LLaVA15_13b(LLaVABase):
    size = 13
