"""Ternary and int8 weights for the learned head. ROADMAP 3c.

A matrix multiply against ternary weights {-g, 0, +g} is additions and subtractions plus one scale,
which is the reason to want it: cheaper, and integer all the way down, so it can be computed
exactly (BitNet b1.58, arXiv 2402.17764; matmul-free LMs, arXiv 2406.02528).

Two ways to get there, measured against each other because they behave very differently:

* **post-training** (``quantize_``): round a float-trained model's weights after the fact. Cheap,
  and usually destructive at ternary width, because nothing in training kept the weights near
  three values.
* **quantisation-aware** (``ternarize_qat``): every forward uses the ternary weights; the
  straight-through estimator passes the gradient to the latent float weights (Bengio et al.
  2013, as BitNet uses it). The model learns under the constraint it will be deployed with.

Targets are every 2-D parameter named ``*weight`` outside embeddings and norms. That covers
``nn.Linear`` and attention's packed ``in_proj_weight``, which is not an ``nn.Linear``.
Parametrisation reaches both without touching the model code. Activations stay float here, so this
measures whether the *weights* can be ternary. Integer activations are a separate step.
"""

from __future__ import annotations

from typing import Callable, Iterator, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


def ternary(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 absmean rounding: per-tensor scale g = mean|w|, values in {-g, 0, +g}."""
    g = w.abs().mean().clamp_min(1e-8)
    return (w / g).round().clamp(-1, 1) * g


def int8(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-row int8: the reference point a ternary result is read against."""
    s = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    return (w / s).round().clamp(-127, 127) * s


def targets(model: nn.Module) -> Iterator[Tuple[nn.Module, str]]:
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)) or "Norm" in type(module).__name__:
            continue
        for name, p in list(module.named_parameters(recurse=False)):
            if name.endswith("weight") and p.dim() == 2:
                yield module, name


def coverage(model: nn.Module) -> float:
    """Share of all parameters that ``targets`` quantises."""
    hit = sum(getattr(m, n).numel() for m, n in targets(model))
    return hit / max(1, sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def quantize_(model: nn.Module, fn: Callable[[torch.Tensor], torch.Tensor]) -> nn.Module:
    """Post-training: overwrite every target weight with its quantised value, in place."""
    for m, n in list(targets(model)):
        getattr(m, n).copy_(fn(getattr(m, n)))
    return model


class _TernarySTE(nn.Module):
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + (ternary(w) - w).detach()      # forward ternary, backward identity


def ternarize_qat(model: nn.Module) -> nn.Module:
    """Quantisation-aware: every target weight is read through the ternary STE from now on."""
    for m, n in list(targets(model)):
        parametrize.register_parametrization(m, n, _TernarySTE())
    return model
