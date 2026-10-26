"""UNI ViT attention instrumentation for Generic Attention Explainability (GAE).

Build the same timm architecture used for feature extraction, then replace
its attention modules while preserving their weights. Explicit attention
matrices and retained gradients support relevance propagation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Attention(nn.Module):
    """Multi-head self-attention that stores its attention matrix.

    A drop-in replacement for timm's ViT ``Attention`` on the un-fused path.
    Same submodule layout (``qkv`` / ``q_norm`` / ``k_norm`` / ``attn_drop`` /
    ``proj`` / ``proj_drop``), so trained weights load straight across with
    :meth:`torch.nn.Module.load_state_dict`. The softmaxed matrix is kept on
    ``self.attention_map`` with retained gradients so GAE can read
    both it and ``∂loss/∂A``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        # GAE always runs un-fused so the matrix is available.
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Retain attention for GAE; disable storage during probability inference.
        self.store_attention: bool = True
        self.attention_map: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Accept the keyword arguments forwarded by timm transformer blocks.
        if is_causal:
            raise NotImplementedError("Causal attention is not used by ViT / GAE")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Retain gradients only for the relevance pass.
        if self.store_attention:
            self.attention_map = attn
            if attn.requires_grad:
                attn.retain_grad()

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _unwrap_state_dict(obj):
    """Unwrap a checkpoint saved under a common wrapping key.

    Mirrors ``features_extraction.model._unwrap_state_dict`` so a single UNI
    checkpoint loads identically whether it feeds the feature extractor or
    this instrumented backbone.
    """
    if not isinstance(obj, dict):
        return obj
    for key in ("state_dict", "model_state_dict", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def get_uni_vit(
    img_size: int = 224,
    patch_size: int = 16,
    init_values: float = 1e-5,
    dynamic_img_size: bool = True,
) -> nn.Module:
    """Create a stock UNI-style ViT-L/16 (timm's own blocks, no instrumentation).

    Identical to ``features_extraction.model.load_uni_vit``'s architecture, so
    the same checkpoint loads with ``strict=True``.
    """
    import timm

    return timm.create_model(
        "vit_large_patch16_224",
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        num_classes=0,
        dynamic_img_size=dynamic_img_size,
    )


def instrument_attention(model: nn.Module) -> nn.Module:
    """Swap every block's ``attn`` for the instrumented :class:`Attention`.

    Reads each block's existing attention config (dim, heads, biases, optional
    qk-norm) and copies the trained weights across, so the model stays
    numerically identical but now records per-block attention matrices. Any
    extra submodules a newer timm may add to its own ``Attention`` (e.g. a
    scale-norm that UNI does not use) are simply not present after the swap.
    """
    for blk in model.blocks:
        old = blk.attn
        dim = old.qkv.in_features
        qkv_bias = old.qkv.bias is not None
        qk_norm = not isinstance(getattr(old, "q_norm", nn.Identity()), nn.Identity)
        new = Attention(
            dim,
            num_heads=old.num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=getattr(old.attn_drop, "p", 0.0),
            proj_drop=getattr(old.proj_drop, "p", 0.0),
        )
        # Preserve exact scaling/geometry, then copy overlapping weights.
        new.scale = old.scale
        new.head_dim = old.head_dim
        new.load_state_dict(old.state_dict(), strict=False)
        new.to(next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        blk.attn = new
    return model


def disable_fused_attention(model: nn.Module) -> None:
    """Belt-and-braces: force every (instrumented) block onto the un-fused path."""
    for blk in model.blocks:
        blk.attn.fused_attn = False


def set_store_attention(model: nn.Module, flag: bool) -> None:
    """Toggle attention-matrix storage on every instrumented block.

    Off = lightweight forward for the probability/triage pass. On = GAE pass
    (default). When turning off, any previously stored matrices are released.
    """
    for blk in model.blocks:
        blk.attn.store_attention = flag
        if not flag:
            blk.attn.attention_map = None


def build_uni_vit_gae(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    *,
    img_size: int = 224,
    patch_size: int = 16,
    init_values: float = 1e-5,
    strict: bool = True,
) -> nn.Module:
    """Build the UNI ViT, load ``checkpoint_path``, and instrument its attention.

    The checkpoint is loaded into a stock timm model (``strict`` matching
    guaranteed), *then* the attention modules are swapped — so version-specific
    block kwargs never reach a hand-written constructor. Returns an eval-mode,
    float32 model on ``device`` whose ``block.attn.attention_map`` is populated
    on every forward. Single precision is deliberate: the paper removes the head
    sigmoid specifically so GAE is numerically stable in float32.
    """
    model = get_uni_vit(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
    )
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(state)
    model.load_state_dict(state, strict=strict)
    instrument_attention(model)
    disable_fused_attention(model)
    model.eval().float().to(device)
    log.info("Loaded GAE UNI backbone from %s (attention instrumented, un-fused)", checkpoint_path)
    return model
