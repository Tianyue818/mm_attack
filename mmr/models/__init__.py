import os
import sys
import importlib
from typing import Mapping

from .base import MixedPrompt, MixedPrompts, MultiModalModelBase
from .utils import disable_random_init_context


MODEL_LIST: Mapping[str, Mapping[str, str]] = {
    'llama2': {'local': 'llm.Llama2'},
    'minigpt4': {'local': 'minigpt.MiniGPT4'},
    'minigpt5': {'local': 'minigpt.MiniGPT5'},
    'mplug': {'local': 'mplug.mplug.Mplug', 'external': 'mPLUG-Owl/mPLUG-Owl'},
    'mplug2': {'local': 'mplug.mplug2.Mplug2', 'external': 'mPLUG-Owl/mPLUG-Owl2'},
    'llava15-7b': {'local': 'llava.LLaVA15_7b', 'external': 'LLaVA'},
    'llava15-7b-lora': {'local': 'llava.LLaVA15_7b_LoRA', 'external': 'LLaVA'},
    'llava15-13b': {'local': 'llava.LLaVA15_13b', 'external': 'LLaVA'},
    'llava15-13b-lora': {'local': 'llava.LLaVA15_13b_LoRA', 'external': 'LLaVA'},
    'blip2-flan-t5-xl': {'local': 'lavis.Blip2FlanT5XL'},
    'blip2-flan-t5-xxl': {'local': 'lavis.Blip2FlanT5XXL'},
    'blip2-opt-3b': {'local': 'lavis.Blip2Opt3b'},
    'blip2-opt-7b': {'local': 'lavis.Blip2Opt7b'},
    'blip2-opt-7b-coco': {'local': 'lavis.Blip2Opt7bCoco'},
    'instructblip-vicuna-7b': {'local': 'lavis.InstructBlipVicuna7b'},
    'instructblip-vicuna-13b': {'local': 'lavis.InstructBlipVicuna13b'},
    'instructblip-flan-t5-xl': {'local': 'lavis.InstructBlipFlanT5XL'},
    'instructblip-flan-t5-xxl': {'local': 'lavis.InstructBlipFlanT5XXL'},
}


def get_model(name: str, **kwargs) -> MultiModalModelBase:
    # use module path to import model for lazy loading
    info = MODEL_LIST[name.lower()]
    third_party_root = os.path.join(
        os.path.dirname(__file__), '..', '..', 'third_party')
    if external := info.get('external'):
        third_party_path = os.path.join(third_party_root, external)
        if third_party_path not in sys.path:
            sys.path.append(third_party_path)
    *module_path, model_name = info['local'].split('.')
    module = importlib.import_module('.' + '.'.join(module_path), __package__)
    with disable_random_init_context():
        return getattr(module, model_name)(**kwargs)


__all__ = [
    'MixedPrompt', 'MixedPrompts', 'MultiModalModelBase',
    'MODEL_LIST', 'get_model',
]
