from typing import Optional, Callable, Iterable, Union, Mapping, Any

import torch
from torch import Tensor

from mmr.models import MultiModalModelBase
from mmr.attack.utils import (
    clip, batch_slice, image_slices, update_embedding_slices)


def ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(logits, targets)


def lafeat_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size = logits.shape[0]
    onehot = torch.nn.functional.one_hot(targets, logits.shape[1])
    top1 = logits[torch.arange(batch_size), targets]
    top2 = torch.max((1.0 - onehot) * logits, dim=1).values
    scale = (top1 - top2).detach().unsqueeze(1).clamp_(min=0.1)
    return torch.nn.functional.cross_entropy(logits / scale, targets)


def pgd_attack(
    model: MultiModalModelBase,
    questions: Iterable[str],
    images: Union[Tensor, Iterable[Tensor]],
    answers: Iterable[str],
    epsilon: float = 8 / 255,
    step_size: float = 2 / 255,
    step_schedule: str = 'constant',
    momentum: float = 0.9,
    steps: int = 100,
    generate: int = 0,
    generate_interval: int = 1,
    max_new_tokens: int = 10,
    early_stop: Optional[Callable[[Iterable[str]], bool]] = None,
    loss_func: Callable[[Tensor, Tensor], Tensor] = ce_loss,
) -> Mapping[str, Any]:
    questions, answers = [list(i) for i in [questions, answers]]
    prompts = model.format_prompts(questions, images=images, responses=answers)
    emb_map = model.to_embedding(prompts)
    target_slices = [s[-1] for s in emb_map['slices']]
    params = []
    if not isinstance(images, Tensor):
        images = torch.stack(list(images), 0)
    perts = 2 * epsilon * torch.rand_like(images) - epsilon
    perts = torch.nn.Parameter(perts)
    params.append(perts)
    attack_images = images
    attack_image_slices = image_slices(prompts, emb_map['slices'])
    for i in range(steps):
        attack_images = clip(images, perts, 'linf', epsilon)
        perts.data = attack_images - images
        if generate and i % generate_interval == 0:
            qprompts = model.format_prompts(questions, images=attack_images)
            qprompts = [q for qs in qprompts for q in [qs] * generate]
            output = model.generate(qprompts, max_new_tokens=max_new_tokens)
            for j, o in enumerate(output):
                print(f'{j // generate}.{j % generate}: {o}')
            if early_stop is not None and early_stop(output):
                break
        # The following is identical
        # to model.to_embedding(model.format_prompts()),
        # but we do not use it for extra safety.
        # image_emb_map = model.images_to_embedding(attack_images)
        # emb_map['embs'] = update_embedding_slices(
        #     emb_map['embs'], attack_image_slices, image_emb_maps)
        prompts = model.format_prompts(
            questions, images=attack_images, responses=answers)
        emb_map = model.to_embedding(prompts)
        logits = model.embedding_forward(emb_map).logits
        logits = batch_slice(logits, target_slices, -1)
        targets = batch_slice(emb_map['toks'], target_slices)
        loss = loss_func(logits, targets)
        loss.backward()
        print(f'PGD step {i + 1}: {loss=:.3f}')
        perts.data -= step_size * perts.grad.sign()
        perts.grad = None
    return_dict: Mapping[str, Any] = {
        'prompts': prompts,
        'attack_images': attack_images,
        'perts': perts,
        'attack_image_slices': attack_image_slices,
    }
    return return_dict
