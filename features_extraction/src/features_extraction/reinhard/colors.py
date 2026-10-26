"""Batched RGB and CIE Lab conversion on PyTorch tensors.

Both functions accept ``(3, H, W)`` and ``(B, 3, H, W)`` tensors. Channel mixing
uses einsum and preserves the batch dimension.
"""

import torch

_RGB2XYZ = torch.tensor(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

_XYZ2RGB = torch.tensor(
    [
        [3.2404813432005266, -1.5371515162713185, -0.4985363261688878],
        [-0.9692549499965682, 1.8759900014898907, 0.0415559265582928],
        [0.0556466391351772, -0.2040413383665112, 1.0573110696453443],
    ]
)

_D65_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def _check_chw(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(x)}")
    if x.dim() < 3 or x.shape[-3] != 3:
        raise ValueError(f"{name}: expected (..., 3, H, W), got {tuple(x.shape)}")


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB in [0, 1] to CIE Lab (D65). Shape (..., 3, H, W) preserved.

    L is in [0, 100]; a, b are in roughly [-128, 127].
    """
    _check_chw(rgb, "rgb_to_lab")

    # sRGB -> linear RGB
    lin_rgb = torch.where(
        rgb > 0.04045,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92,
    )

    # Linear RGB -> XYZ, contracting on the channel dim regardless of batch shape.
    mat = _RGB2XYZ.to(device=lin_rgb.device, dtype=lin_rgb.dtype)
    xyz = torch.einsum("ck,...khw->...chw", mat, lin_rgb)

    white = _D65_WHITE.to(device=xyz.device, dtype=xyz.dtype).view(3, 1, 1)
    xyz_n = xyz / white

    threshold = 0.008856
    power = torch.pow(xyz_n.clamp(min=threshold), 1.0 / 3.0)
    scale = 7.787 * xyz_n + 4.0 / 29.0
    f = torch.where(xyz_n > threshold, power, scale)

    fx = f[..., 0, :, :]
    fy = f[..., 1, :, :]
    fz = f[..., 2, :, :]

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.stack([L, a, b], dim=-3)


def lab_to_rgb(lab: torch.Tensor, clip: bool = True) -> torch.Tensor:
    """Inverse of :func:`rgb_to_lab`. Output is sRGB in [0, 1] (clipped by default)."""
    _check_chw(lab, "lab_to_rgb")

    L = lab[..., 0, :, :]
    a = lab[..., 1, :, :]
    b = lab[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = (fy - b / 200.0).clamp(min=0.0)
    f = torch.stack([fx, fy, fz], dim=-3)

    power = torch.pow(f, 3.0)
    scale = (f - 4.0 / 29.0) / 7.787
    xyz_n = torch.where(f > 0.2068966, power, scale)

    white = _D65_WHITE.to(device=lab.device, dtype=lab.dtype).view(3, 1, 1)
    xyz = xyz_n * white

    mat = _XYZ2RGB.to(device=lab.device, dtype=lab.dtype)
    rgb_lin = torch.einsum("ck,...khw->...chw", mat, xyz)

    threshold = 0.0031308
    rgb = torch.where(
        rgb_lin > threshold,
        1.055 * torch.pow(rgb_lin.clamp(min=threshold), 1.0 / 2.4) - 0.055,
        12.92 * rgb_lin,
    )

    if clip:
        rgb = rgb.clamp(0.0, 1.0)
    return rgb
