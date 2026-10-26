"""Select saliency patches with variant, Otsu, percentile, or fixed thresholds.

Variant defaults are 0.01 for NIP, 0.5 for bag-normalized NIP, and 1.0 for GIP.
If a cutoff retains too few patches, selection falls back to Otsu. Otsu operates
on min-max-normalized probabilities and returns a cutoff in the input scale.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: model_type → paper probability threshold. NIP depends on bag-size norm.
PAPER_DEFAULTS = {
    "linear_gated_ratio": 1.0,  # GIP
    "linear_noisy_or": (1e-2, 0.5),  # NIP: (no-norm, with-norm)
}

_PERCENTILE_RE = re.compile(r"^p(\d+(?:\.\d+)?)$", re.IGNORECASE)


@dataclass(frozen=True)
class ResolvedThreshold:
    """The numeric cutoff plus a short description for logs / filenames."""

    value: float
    method: str  # "paper" | "auto" | "percentile" | "fixed"

    def __str__(self) -> str:
        return f"{self.value:.4g} ({self.method})"


def default_threshold(model_type: str, params: Optional[Mapping[str, Any]] = None) -> float:
    """The paper's fixed probability threshold for a head type."""
    params = params or {}
    entry = PAPER_DEFAULTS.get(model_type)
    if entry is None:
        log.warning("No paper threshold for model_type %r; defaulting to 0.5", model_type)
        return 0.5
    if isinstance(entry, tuple):
        no_norm, with_norm = entry
        return with_norm if params.get("normalize_by_bag_size") else no_norm
    return float(entry)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu cutoff on the min-max-normalised ``values``, mapped back to raw scale.

    Returns a threshold in the original ``values`` range. Falls back to the
    median for degenerate (constant / tiny) distributions.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    norm = (v - lo) / (hi - lo)

    hist, edges = np.histogram(norm, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(np.median(v))

    w0 = np.cumsum(hist)
    w1 = total - w0
    centers = (edges[:-1] + edges[1:]) / 2.0
    csum = np.cumsum(hist * centers)
    mean_total = csum[-1]

    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        return float(np.median(v))
    mean0 = np.where(w0 > 0, csum / np.maximum(w0, 1e-12), 0.0)
    mean1 = np.where(w1 > 0, (mean_total - csum) / np.maximum(w1, 1e-12), 0.0)
    between = w0 * w1 * (mean0 - mean1) ** 2
    between[~valid] = -1.0
    t_norm = float(centers[int(np.argmax(between))])
    return lo + t_norm * (hi - lo)


def resolve_threshold(
    spec: str,
    probs: np.ndarray,
    *,
    model_type: str,
    params: Optional[Mapping[str, Any]] = None,
) -> ResolvedThreshold:
    """Turn a CLI ``--threshold`` spec into a concrete cutoff.

    Args:
        spec: ``"paper"``, ``"auto"``, ``"pNN"`` (percentile), or a float string.
        probs: the patch probabilities the threshold will be applied to
            (used by ``auto`` / percentile).
        model_type, params: head info for the ``paper`` default.
    """
    s = str(spec).strip().lower()

    if s == "paper":
        return ResolvedThreshold(default_threshold(model_type, params), "paper")

    if s == "auto":
        return ResolvedThreshold(otsu_threshold(probs), "auto")

    m = _PERCENTILE_RE.match(s)
    if m:
        q = float(m.group(1))
        val = float(np.percentile(np.asarray(probs, dtype=np.float64), q)) if len(probs) else 0.0
        return ResolvedThreshold(val, f"percentile p{q:g}")

    try:
        return ResolvedThreshold(float(s), "fixed")
    except ValueError as e:
        raise ValueError(
            f"Invalid --threshold {spec!r}; expected 'paper', 'auto', 'pNN', or a float."
        ) from e


def select_mask(
    probs: np.ndarray,
    threshold: ResolvedThreshold,
    *,
    fallback_probs: Optional[np.ndarray] = None,
    min_keep: int = 1,
) -> Tuple[np.ndarray, ResolvedThreshold]:
    """Boolean mask of patches with ``prob >= threshold.value``.

    If the cutoff keeps fewer than ``min_keep`` patches (e.g. the degenerate
    GIP=1.0 paper value), fall back to an Otsu threshold on
    ``fallback_probs`` (defaults to ``probs``) so we never emit an empty map.
    Returns the (possibly updated) mask and threshold actually used.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    mask = probs >= threshold.value
    if int(mask.sum()) >= min_keep or probs.size == 0:
        return mask, threshold

    src = fallback_probs if fallback_probs is not None else probs
    fb = ResolvedThreshold(otsu_threshold(src), "auto-fallback")
    log.warning(
        "Threshold %s kept %d/%d patches; falling back to %s",
        threshold,
        int(mask.sum()),
        probs.size,
        fb,
    )
    return probs >= fb.value, fb
