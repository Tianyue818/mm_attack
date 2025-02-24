from typing import Mapping, Any

import torch
import peft
from huggingface_hub import hf_hub_download
from torch import Tensor
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel

from .minigpt5 import MiniGPT5


class TinyGPT5(MiniGPT5):
    image_size = 448
    template = ['<s>[INST] ', ' [/INST] ']
    language_model_path = 'susnato/phi-2'
    qformer_url = (
        'https://storage.googleapis.com'
        '/sfr-vision-language-research/LAVIS/models/BLIP2'
        '/blip2_pretrained_flant5xxl.pth')
    qformer_path = './pretrained/minigpt4/blip2_pretrained_flant5xxl.pth'
    lora_model_path = ('Tyrannosaurus/TinyGPT-V', 'TinyGPT-V_for_Stage4.pth')
    bert_model_path = 'bert-base-uncased'
    num_query_tokens = 32

    def __init__(self, **hf_kwargs: Mapping[str, Any]):
        super().__init__(**hf_kwargs)
        self.Qformer, self.query_tokens = self._init_qformer()
        self.eval().requires_grad_(False)

    def _init_qformer(self):
        encoder_config = BertConfig.from_pretrained(self.bert_model_path)
        encoder_config.encoder_width = self.vision_model[0].num_features
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = self.num_query_tokens
        qformer = BertLMHeadModel(config=encoder_config)
        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None
        for layer in qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        for param in qformer.parameters():
            param.requires_grad = False
        try:
            state = torch.load(self.qformer_path)['model']
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Checkpoint {self.qformer_path!r} not found, '
                f'please download it from {self.qformer_url!r}.'
            ) from e
        qstate = {k.replace('Qformer.', ''): v for k, v in state.items()}
        qformer.load_state_dict(qstate)
        self.vision_model[1].load_state_dict(state['vision_model']['ln_vision'])
        query_tokens = torch.nn.Parameter(
            state['query_tokens'], requires_grad=False)
        return qformer, query_tokens

    def _init_projection(self):
        lora_config = peft.LoraConfig(
            r=64, lora_alpha=16, bias='none', task_type='CAUSAL_LM',
            target_modules=['q_proj', 'v_proj'])
        self.language_model = peft.get_peft_model(
            self.language_model, lora_config)
        cache_file = hf_hub_download(*self.lora_model_path)
        lora_state = torch.load(cache_file, map_location='cpu')
        # model.load_state_dict(state, strict=False)
        peft.set_peft_model_state_dict(self.language_model, state['model'])
        self.qformer, self.query_tokens = self._init_qformer()
        # Initialize the projection layer
        proj = torch.nn.Sequential(
            torch.nn.Linear(self.qformer.config.hidden_size, 4096),
            # TinyGPT-V ridiuclously
            # did not use an activation function
            # between the two linear layers.
            # torch.nn.ReLU(),
            torch.nn.Linear(4096, self.language_model.config.hidden_size),
        )
        proj_state = {
            '1.weight': lora_state['model']['llama_proj.weight'],
            '1.bias': lora_state['model']['llama_proj.bias'],
            '2.weight': lora_state['model']['llama_proj2.weight'],
            '2.bias': lora_state['model']['llama_proj2.bias'],
        }
        proj.load_state_dict(proj_state, strict=True)
        return self.to_default_dtype(proj)

    def images_to_embedding(self, image: Tensor) -> Tensor:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = self.image_normalizer(image)
        features = self.vision_model(image)
        attns = torch.ones_like(features.size()[:-1], dtype=torch.long)
        query_tokens = self.query_tokens.expand(features.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens, encoder_hidden_states=features,
            encoder_attention_mask=attns, return_dict=True)
        return self.projection(query_output.last_hidden_state)
