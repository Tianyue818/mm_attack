import functools
from typing import Optional, Mapping, Any, Iterable, Union, List

import torch
from torch import Tensor
import torchvision as tv

from ..llm import Llama2
from ..utils import cached_load
from .vit import VisionTransformer, interpolate_pos_embed


class MiniGPT4(Llama2):
    vision_model: torch.nn.Module
    image_size = 224
    template = [
        '<s>Give the following image: <Img>ImageContent</Img>. '
        'You will be able to see the image once I provide it to you. '
        'Please answer my questions.<s>[INST] ',
        ' [/INST] '
    ]
    vision_model_path = (
        'https://storage.googleapis.com'
        '/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth')
    # vision_model_path = './pretrained/minigpt4/eva_vit_g.pth'
    # language_model_path = './pretrained/minigpt4/Llama-2-7b-chat-hf'
    language_model_path = 'meta-llama/Llama-2-7b-chat-hf'
    # blip2_model_path = './pretrained/minigpt4/blip2_minigpt4_llama2_7b.pth'
    llama_proj_model_path = './pretrained/minigpt4/llama_proj.pth'

    def __init__(self, **hf_kwargs: Mapping[str, Any]):
        super().__init__(**hf_kwargs)
        self.image_normalizer = tv.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        self.vision_model = self._init_vision_transformer()
        self.projection = self._init_projection()
        self.eval().requires_grad_(False)

    def _init_vision_transformer(self) -> torch.nn.Sequential:
        LayerNorm = functools.partial(torch.nn.LayerNorm, eps=1e-6)
        vision_model = VisionTransformer(
            img_size=self.image_size, patch_size=14, use_mean_pooling=False,
            embed_dim=1408, depth=39, num_heads=1408 // 88, mlp_ratio=4.3637,
            qkv_bias=True, drop_path_rate=0, norm_layer=LayerNorm,
            use_checkpoint=False)
        state = cached_load(self.vision_model_path)
        interpolate_pos_embed(vision_model, state)
        vision_model.load_state_dict(state, strict=False)
        lnorm = torch.nn.LayerNorm(vision_model.num_features)
        seq = torch.nn.Sequential(vision_model, lnorm)
        return self.to_default_dtype(seq)

    def _init_projection(self):
        proj = torch.nn.Linear(
            self.vision_model[0].num_features * 4,
            self.language_model.config.hidden_size)
        # blip2_state = torch.load(self.blip2_model_path)
        # proj_state = {
        #     'weight': blip2_state['model']['llama_proj.weight'],
        #     'bias': blip2_state['model']['llama_proj.bias'],
        # }
        # proj.load_state_dict(proj_state, strict=False)
        proj_state = torch.load(self.llama_proj_model_path)
        proj.load_state_dict(proj_state, strict=True)
        return self.to_default_dtype(proj)

    def format_prompt(
        self, question: str, suffix: Optional[str] = None,
        image: Optional[Tensor] = None, response: Optional[str] = None,
    ) -> Iterable[Union[str, Tensor]]:
        t1, t2 = self.template
        prompt: List[Union[str, Tensor]] = [t1, question]
        if suffix is not None:
            prompt.append(suffix)
        if image is not None:
            prompt += ['<Img>', image, '</Img>']
        prompt.append(t2)
        if response is not None:
            prompt.append(response)
        return prompt

    def images_to_embedding(self, image: Tensor) -> Tensor:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = self.image_normalizer(image)
        features = self.vision_model(image)
        features = features[:, 1:, :]
        bs, pn, hs = features.shape
        features = features.view(bs, int(pn / 4), int(hs * 4))
        return self.projection(features)
