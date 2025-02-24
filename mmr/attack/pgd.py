import math
from typing import Literal, Optional

import torch
from torch import nn, Tensor

from .utils import clip, tslice, ThreatModel
from .session import AttackSession
from .base import BaseAttack


StepSizeSchedule = Literal['constant', 'linear', 'cosine']


class PGDAttack(BaseAttack):
    iterations: int
    step_size: Optional[float]
    step_size_schedule: StepSizeSchedule
    momentum: float

    def __init__(
        self, iterations: int, early_stop: bool = True,
        step_size: Optional[float] = None, momentum: float = 0,
        step_size_schedule: StepSizeSchedule = 'constant'
    ) -> None:
        super().__init__(early_stop)
        self.iterations = iterations
        self.step_size = step_size
        self.momentum = momentum
        self.step_size_schedule = step_size_schedule

    def grad_schedule(
        self, epsilon: float, schedule: float, grad: Tensor
    ) -> Tensor:
        if self.step_size_schedule == 'constant':
            s = self.step_size
        elif self.step_size_schedule == 'linear':
            s = 2 * epsilon * (1 - schedule)
        elif self.step_size_schedule == 'cosine':
            s = epsilon * (1 + math.cos(schedule * math.pi))
        else:
            raise ValueError(
                f'Unknown step size schedule {self.step_size_schedule!r}.')
        return s * grad.sign()

    def update_func(
        self, threat: ThreatModel, epsilon: float, schedule: float,
        xai: Tensor, xai2: Tensor, xgi: Tensor, xw: Tensor
    ) -> Tensor:
        xan = xai + self.grad_schedule(epsilon, schedule, xgi)
        xan = clip(xan, xw, threat, epsilon)
        mmt = self.momentum if schedule > 0 else 0
        xai = xai + (1 - mmt) * (xan - xai) + mmt * (xai - xai2)
        return clip(xai, xw, threat, epsilon)

    def attack_batch(
        self, session: AttackSession, indices: Tensor,
        images: Tensor, labels: Tensor, adv_images: Optional[Tensor] = None
    ) -> Tensor:
        s = session
        batch_size = images.shape[0]
        ii = indices
        xi = images
        if adv_images is None:
            adv_images = images
        xa, xai = adv_images.clone(), adv_images.clone()
        xai2 = xai.detach()
        li = labels.clone()
        success = torch.zeros(batch_size, device=xa.device).bool()
        for i in range(self.iterations):
            self.iteration = i
            xai.requires_grad_()
            oi = s.model(xai)
            s.on_step(ii, oi, li, xi, xai)
            if not self.early_stop:
                loss = self.loss_func(s, ii, oi, li, xi, xai).mean()
                xgi = torch.autograd.grad(loss, xai)[0]
            else:
                si = (oi.max(1)[1] != li).detach()
                xa[~success] = xai.detach()
                success[~success] = si
                loss = self.loss_func(s, ii, oi, li, xi, xai).mean()
                oi, li, ii, xai2, xi = tslice((oi, li, ii, xai2, xi), ~si)
                xgi = torch.autograd.grad(loss, xai)[0]
                xgi, xai = tslice((xgi, xai), ~si)
            xai2, xai = xai.detach(), self.update_func(
                s.threat_name, s.epsilon,
                i / self.iterations, xai, xai2, xgi, xi).detach()
        if self.early_stop:
            xa[~success] = xai.detach()
        else:
            xa = xai.detach()
        return xa


class LafeatAttack(PGDAttack):
    def __init__(
        self, iterations: int, early_stop: bool = True,
        momentum: float = 0.75, temperature: float = 1.0
    ) -> None:
        super().__init__(
            iterations, early_stop, momentum=momentum,
            step_size_schedule='cosine')
        self.temperature = temperature

    def loss_func(
        self, session: AttackSession,
        indices: Tensor, outputs: Tensor, labels: Tensor,
        images: Tensor, adv_images: Tensor
    ) -> Tensor:
        batch_size = outputs.shape[0]
        onehot = nn.functional.one_hot(labels, session.num_classes)
        top1 = outputs[range(batch_size), labels]
        top2 = torch.max((1.0 - onehot) * outputs, dim=1).values
        scale = (top1 - top2).detach().unsqueeze(1).clamp_(min=0.1)
        outputs = outputs / scale / self.temperature
        return nn.functional.cross_entropy(outputs, labels, reduction='none')
