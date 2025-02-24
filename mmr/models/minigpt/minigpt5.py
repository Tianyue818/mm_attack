import torch
import peft

from .minigpt4 import MiniGPT4


class MiniGPT5(MiniGPT4):
    image_size = 448
    template = ['<s>[INST] ', ' [/INST] ']
    lora_model_path = './pretrained/minigpt5/minigptv2_checkpoint.pth'
    lora_model_url = 'https://drive.google.com/u/0/uc?id=1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl'

    def _init_projection(self):
        # Initialize a LoRA on top of the Llama 2 model.
        # model = peft.prepare_model_for_int8_training(model)
        lora_config = peft.LoraConfig(
            r=64, lora_alpha=16, bias='none', task_type='CAUSAL_LM',
            target_modules=['q_proj', 'v_proj'])
        self.language_model = peft.get_peft_model(
            self.language_model, lora_config)
        try:
            state = torch.load(self.lora_model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Checkpoint {self.llama_lora_model_path!r} not found, '
                f'please download it from {self.lora_model_url!r}.'
            ) from e
        # model.load_state_dict(state, strict=False)
        peft.set_peft_model_state_dict(self.language_model, state['model'])
        # Initialize the projection layer
        proj = torch.nn.Linear(
            self.vision_model[0].num_features * 4,
            self.language_model.config.hidden_size)
        proj_state = {
            'weight': state['model']['llama_proj.weight'],
            'bias': state['model']['llama_proj.bias'],
        }
        proj.load_state_dict(proj_state, strict=True)
        return self.to_default_dtype(proj)

    def tokens_to_embedding(self, ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.base_model.model.model.embed_tokens(ids)
