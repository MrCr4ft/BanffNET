"""Generic Attention Explainability (Chefer et al., ICCV 2021).

Positive attention-gradient products are averaged across heads and propagated
as ``R = R + A_bar @ R``, starting from the identity. The CLS-to-patch row is
min-max normalized per patch and bilinearly upsampled. The target model must
emit logits and run in float32 for stable gradients.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

log = logging.getLogger(__name__)

#: Accumulate relevance in float32 for numerical stability.
ACC_DTYPE = torch.float32


def _avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Positive part of the head-averaged ``A ⊙ ∂loss/∂A``.

    Args:
        cam:  attention matrix ``A``     (B, H, N, N).
        grad: its gradient ``∂loss/∂A``  (B, H, N, N).

    Returns:
        ``Ā`` of shape (B, N, N).
    """
    prod = (cam.to(ACC_DTYPE) * grad.to(ACC_DTYPE)).clamp_min_(0)
    return prod.mean(dim=1)


def _upscale_relevance(rel: torch.Tensor, grid: int, out_size: int) -> torch.Tensor:
    """Reshape the CLS→patch row to ``grid×grid``, min-max norm, bilinear upsample.

    Args:
        rel:      CLS→patch relevance (B, grid*grid).
        grid:     tokens per side (14 for 224px @ patch 16).
        out_size: output spatial size (224).

    Returns:
        (B, 1, out_size, out_size) in [0, 1].
    """
    B = rel.size(0)
    rel = rel.view(B, 1, grid, grid)
    rel = F.interpolate(rel, size=(out_size, out_size), mode="bilinear", align_corners=False)
    rel = rel.flatten(1)
    mn = rel.min(dim=1, keepdim=True).values
    mx = rel.max(dim=1, keepdim=True).values
    rel = (rel - mn) / (mx - mn + 1e-6)
    return rel.view(B, 1, out_size, out_size)


@torch.inference_mode(False)
def compute_relevance(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    out_size: int = 224,
) -> torch.Tensor:
    """GAE relevance maps for a batch of preprocessed patches.

    Args:
        model: an :class:`ExplainableBanffModel` (or any module exposing
            ``.blocks`` whose ``blk.attn.attention_map`` is populated on
            forward) that returns a per-patch **logit** of shape (B, 1).
        inputs: preprocessed patch tensor (B, 3, H, W), float32.
        out_size: spatial size of the returned maps.

    Returns:
        (B, 1, out_size, out_size) relevance maps in [0, 1], on ``inputs``'s
        device.
    """
    model.zero_grad(set_to_none=True)

    # Probability inference may have disabled storage on the shared backbone.
    for blk in model.blocks:
        blk.attn.store_attention = True

    with torch.enable_grad():
        logits = model(inputs)  # (B, 1) — sigmoid removed
        # Independent patches share one backward pass by summing their targets.
        loss = logits.sum()

        attn_maps = [blk.attn.attention_map for blk in model.blocks]
        grads = torch.autograd.grad(loss, attn_maps, retain_graph=False, allow_unused=True)

    num_tokens = attn_maps[0].shape[-1]
    R = torch.eye(num_tokens, dtype=ACC_DTYPE, device=inputs.device)
    R = R.expand(inputs.size(0), -1, -1).clone()

    for cam, grad in zip(attn_maps, grads):
        if grad is None:
            continue
        a_bar = _avg_heads(cam, grad)
        R = R + a_bar @ R

    # First row = information flow from CLS to every patch token (drop CLS self).
    relevance = R[:, 0, 1:]
    grid = int(round((relevance.size(1)) ** 0.5))

    # Release the stored matrices so they don't pin graph memory between batches.
    for blk in model.blocks:
        blk.attn.attention_map = None

    return _upscale_relevance(relevance, grid=grid, out_size=out_size)


def saliency_for_patches(
    model: nn.Module,
    preprocess,
    patches: List[np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 8,
    out_size: int = 224,
    progress: bool = True,
    desc: str = "GAE saliency",
) -> np.ndarray:
    """Compute GAE saliency for a list of raw (H, W, 3) uint8 patches.

    Preprocessing is delegated to ``preprocess`` (a
    :class:`features_extraction.BatchPreprocess`), so stain normalisation and
    ImageNet stats are identical to the feature-extraction / prediction path.

    Returns:
        (N, out_size, out_size) float32 saliency maps in [0, 1].
    """
    n = len(patches)
    out = np.zeros((n, out_size, out_size), dtype=np.float32)
    if n == 0:
        return out

    rng = range(0, n, batch_size)
    if progress:
        rng = tqdm(
            rng,
            desc=desc,
            unit="batch",
            leave=False,
            total=(n + batch_size - 1) // batch_size,
        )

    for start in rng:
        end = min(start + batch_size, n)
        batch = np.stack(patches[start:end], axis=0)  # (b, H, W, 3) uint8
        x = preprocess(batch).to(device=device, dtype=torch.float32)
        rel = compute_relevance(model, x, out_size=out_size)  # (b, 1, S, S)
        out[start:end] = rel.squeeze(1).detach().cpu().numpy()
        del x, rel
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out
