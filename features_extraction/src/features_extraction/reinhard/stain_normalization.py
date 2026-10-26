"""Reinhard stain normalization for ``(3, H, W)`` or ``(B, 3, H, W)`` tensors.

Inputs are in ``[0, 1]``. Reference Lab statistics are registered buffers that
move with the module's device.
"""

import typing
from pathlib import Path

import torch
import yaml

from .colors import lab_to_rgb, rgb_to_lab


class Reinhard(torch.nn.Module):
    """Reinhard color transfer.

    Two variants are implemented:

    * Standard (default): per-sample affine match of all three Lab channels to
      the reference mean/std.
    * Modified (``apply_modified_reinhard=True``): only the L channel is
      rescaled (around the *source* mean), and the a/b channels are either
      left untouched (if the source contrast is below the reference) or
      mean-shifted to the reference (if the source contrast is above).

    Parameters
    ----------
    lab_reference_mean, lab_reference_std
        Length-3 sequences with the reference L, a, b statistics.
    apply_modified_reinhard
        Use the modified variant.
    eps
        Numerical floor for standard-deviation divisions.
    """

    def __init__(
        self,
        lab_reference_mean: typing.Sequence[float],
        lab_reference_std: typing.Sequence[float],
        apply_modified_reinhard: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        ref_mean = torch.as_tensor(lab_reference_mean, dtype=torch.float32).view(1, 3, 1, 1)
        ref_std = torch.as_tensor(lab_reference_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("lab_reference_mean", ref_mean)
        self.register_buffer("lab_reference_std", ref_std)
        self.apply_modified_reinhard = bool(apply_modified_reinhard)
        self.eps = float(eps)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Accept (3, H, W), (H, W, 3) by permute, or (B, 3, H, W).
        single = img.dim() == 3
        if single:
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1).contiguous()
            img = img.unsqueeze(0)
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f"Reinhard expects (3,H,W) or (B,3,H,W); got {tuple(img.shape)}")

        lab = rgb_to_lab(img)  # (B, 3, H, W)
        src_mean = lab.mean(dim=(-2, -1), keepdim=True)  # (B, 3, 1, 1)
        src_std = lab.std(dim=(-2, -1), keepdim=True, unbiased=True)  # (B, 3, 1, 1)

        ref_mean = self.lab_reference_mean.to(dtype=lab.dtype)
        ref_std = self.lab_reference_std.to(dtype=lab.dtype)

        if not self.apply_modified_reinhard:
            lab = (lab - src_mean) / src_std.clamp_min(self.eps) * ref_std + ref_mean
        else:
            # q is a per-sample scalar derived from the L channel only.
            q = (ref_std[:, 0:1] - src_std[:, 0:1]) / ref_std[:, 0:1].clamp_min(self.eps)
            q_pos = q > 0
            q_mul = torch.where(q_pos, 1.0 + q, torch.full_like(q, 1.05))

            L = (lab[:, 0:1] - src_mean[:, 0:1]) * q_mul + src_mean[:, 0:1]
            a_shift = lab[:, 1:2] - src_mean[:, 1:2] + ref_mean[:, 1:2]
            b_shift = lab[:, 2:3] - src_mean[:, 2:3] + ref_mean[:, 2:3]
            a = torch.where(q_pos, lab[:, 1:2], a_shift)
            b = torch.where(q_pos, lab[:, 2:3], b_shift)
            lab = torch.cat([L, a, b], dim=1)

        rgb = lab_to_rgb(lab)
        if single:
            rgb = rgb.squeeze(0)
        return rgb


def get_reinhard_normalizer(reinhard_config: typing.Mapping[str, typing.Any]) -> Reinhard:
    """Build a :class:`Reinhard` from a configuration mapping."""
    return Reinhard(
        lab_reference_mean=[
            reinhard_config["l_mean"],
            reinhard_config["a_mean"],
            reinhard_config["b_mean"],
        ],
        lab_reference_std=[
            reinhard_config["l_std"],
            reinhard_config["a_std"],
            reinhard_config["b_std"],
        ],
        apply_modified_reinhard=reinhard_config.get("apply_modified_reinhard", False),
    )


def load_reinhard_from_yaml(path: typing.Union[str, Path]) -> Reinhard:
    """Convenience loader: parse a YAML file and return a configured :class:`Reinhard`."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return get_reinhard_normalizer(cfg)
