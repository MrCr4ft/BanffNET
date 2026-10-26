"""Compose a frozen UNI ViT and a single BanffNET lesion head for GAE.

``forward`` returns a per-patch saliency target without sigmoid saturation;
``predict_proba`` returns probabilities for patch selection and weighting.
Head definitions and checkpoint formats come from ``banffnet.models``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from banffnet.models import MODELS, LinearGatedRatio, LinearNoisyOR

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Human-readable variant label per head type (matches the paper's NIP/GIP).
VARIANT = {
    "linear_noisy_or": "NIP",
    "linear_gated_ratio": "GIP",
}

#: GIP saliency targets — which scalar GAE differentiates for a gated head.
#:
#: * ``classifier``   — ``classifier(gate·h)`` logit: "where does this patch
#:   look like the lesion?" (paper default).
#: * ``gate``         — ``gate(h)`` logit: the model's native per-patch
#:   attention — "what is the model focusing on?".
#: * ``contribution`` — ``log(gate·p) = -softplus(-gate) - softplus(-class)``:
#:   the patch's weighted contribution to the gated-ratio bag score, in a
#:   softplus (sigmoid-stable) log form.
#: * ``both``         — emit ``classifier`` and ``gate`` maps side by side.
GIP_TARGETS = ("classifier", "gate", "contribution", "both")


class ExplainableBanffModel(nn.Module):
    """Backbone + a single lesion head, exposing a sigmoid-free logit for GAE."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        lesion: str,
        model_type: str,
        device: Union[str, torch.device] = "cuda",
        gip_target: str = "classifier",
        gip_detach_gate: bool = False,
    ):
        super().__init__()
        if gip_target not in GIP_TARGETS:
            raise ValueError(f"gip_target must be one of {GIP_TARGETS}, got {gip_target!r}")
        self.backbone = backbone
        self.head = head
        self.lesion = lesion
        self.model_type = model_type
        self.variant = VARIANT.get(model_type, model_type)
        self.gip_target = gip_target
        self.gip_detach_gate = bool(gip_detach_gate)
        # Set by the pipeline before a saliency pass to pick the GIP target
        # for that pass (used by 'both'); None → the configured default.
        self.active_target: Optional[str] = None
        self._device = torch.device(device)
        self.to(self._device).float().eval()

    @property
    def is_gip(self) -> bool:
        """Whether this lesion uses a gated (GIP) head."""
        return isinstance(self.head, LinearGatedRatio)

    def saliency_targets(self) -> List[str]:
        """The saliency map(s) to produce for this lesion.

        NIP always yields a single (untagged) classifier map. GIP yields the
        configured target, or both ``classifier`` and ``gate`` for ``both``.
        """
        if not self.is_gip:
            return ["classifier"]
        return ["classifier", "gate"] if self.gip_target == "both" else [self.gip_target]

    @classmethod
    def from_config(
        cls,
        *,
        uni_checkpoint: PathLike,
        banff_config: PathLike,
        lesion: str,
        device: Union[str, torch.device] = "cuda",
        backbone: nn.Module | None = None,
        gip_target: str = "classifier",
        gip_detach_gate: bool = False,
    ) -> "ExplainableBanffModel":
        """Build for one lesion from a UNI checkpoint and a BanffNET YAML config.

        Pass ``backbone`` to reuse an already-loaded instrumented ViT across
        several lesions (the backbone is frozen and lesion-agnostic).
        ``gip_target`` / ``gip_detach_gate`` are ignored for NIP lesions.
        """
        cfg = _load_banff_config(banff_config)
        if lesion not in cfg:
            raise KeyError(f"Lesion {lesion!r} not in banff config; available: {sorted(cfg)}")
        spec = cfg[lesion]
        model_type = spec["model_type"]
        if model_type not in MODELS:
            raise ValueError(f"Unknown model_type {model_type!r} for lesion {lesion!r}")

        head = MODELS[model_type](**spec["model_params"])
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        head.load_state_dict(state, strict=True)
        head.eval()

        if backbone is None:
            from .backbone import build_uni_vit_gae

            backbone = build_uni_vit_gae(uni_checkpoint, device=device)

        log.info(
            "Explainable model for lesion %r (%s / %s)",
            lesion,
            model_type,
            VARIANT.get(model_type, "?"),
        )
        return cls(
            backbone,
            head,
            lesion=lesion,
            model_type=model_type,
            device=device,
            gip_target=gip_target,
            gip_detach_gate=gip_detach_gate,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def blocks(self):
        """Proxy to the backbone's transformer blocks (used by GAE)."""
        return self.backbone.blocks

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward → pooled CLS features (B, D), with attention captured."""
        return self.backbone(x)

    def instance_logits(self, h: torch.Tensor, target: Optional[str] = None) -> torch.Tensor:
        """Per-patch scalar (sigmoid removed) that GAE differentiates.

        NIP (``LinearNoisyOR``) always returns ``classifier(h)``. GIP
        (``LinearGatedRatio``) returns the requested ``target`` (see
        :data:`GIP_TARGETS`); ``target=None`` falls back to
        :attr:`active_target` then the configured :attr:`gip_target`.
        """
        head = self.head
        if not self.is_gip:
            if isinstance(head, LinearNoisyOR) or hasattr(head, "classifier"):
                return head.classifier(h)
            raise TypeError(f"Head {type(head).__name__} has no linear classifier for GAE")

        # --- GIP ---
        t = target or self.active_target or self.gip_target
        if t == "both":
            t = "classifier"  # 'both' is resolved per-pass by the caller
        gate_logit = head.gate(h)

        if t == "gate":
            return gate_logit

        gate = torch.nan_to_num(torch.sigmoid(gate_logit), nan=0.0)
        if t == "classifier":
            gated = (gate.detach() if self.gip_detach_gate else gate) * h
            return head.classifier(gated)
        if t == "contribution":
            # log(gate · p) with softplus for a sigmoid-stable backward.
            class_logit = head.classifier(gate * h)
            return -F.softplus(-gate_logit) - F.softplus(-class_logit)
        raise ValueError(f"Unknown GIP saliency target {t!r}")

    def forward(self, x: torch.Tensor, target: Optional[str] = None) -> torch.Tensor:
        """Per-patch logit (B, 1) — the scalar GAE differentiates."""
        return self.instance_logits(self.features(x), target)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch probability (B, 1) using the head's own patch-level rule.

        This matches ``banffnet``'s patch-level output exactly, so thresholds
        and weights here mean the same thing as everywhere else in the pipeline.
        """
        return self.proba_from_features(self.features(x))

    @torch.inference_mode()
    def proba_from_features(self, h: torch.Tensor) -> torch.Tensor:
        """Per-patch probability (N, 1) from already-extracted UNI features.

        Lets patch-level runs reuse a features H5 for the (cheap) probability /
        thresholding step while still reading pixels for the (expensive) GAE
        backward.
        """
        return self.head.infer_patch_level(h)

    @property
    def threshold_params(self) -> Dict[str, Any]:
        """Head params relevant to the paper threshold (bag-size normalisation)."""
        return {"normalize_by_bag_size": bool(getattr(self.head, "normalize_by_bag_size", False))}


def _load_banff_config(path: PathLike) -> Dict[str, Dict[str, Any]]:
    with open(path) as f:
        return yaml.safe_load(f)
