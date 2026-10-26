"""Resolve model and staining configuration paths under ``BANFFNET_DATA_ROOT``.

The default root is ``data`` relative to the working directory. Model weights
and staining configurations are supplied separately; see ``INSTALL.md`` for
the expected layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

#: Supported staining identifiers.
STAININGS: Tuple[str, ...] = ("HE", "JONES", "PAS")

#: Model and configuration root, configurable through ``BANFFNET_DATA_ROOT``.
DATA_ROOT: Path = Path(os.environ.get("BANFFNET_DATA_ROOT", "data"))

DEFAULT_PATCH_SIZE: int = 256
DEFAULT_MPP: float = 0.5


@dataclass(frozen=True)
class StainingPaths:
    """Bundle of default paths for one staining."""

    staining: str
    banff_config: Path
    reinhard_config: Path
    binarization_model: Path
    uni_checkpoint: Path


def _make_staining_paths(s: str, root: Path) -> StainingPaths:
    """Construct StainingPaths for a validated, upper-cased staining key."""
    return StainingPaths(
        staining=s,
        banff_config=root / "stainings" / s / "banff_config.yaml",
        reinhard_config=root / "stainings" / s / "reinhard.yaml",
        binarization_model=root / "models" / "binarization" / "mobilenet_v2.pt",
        uni_checkpoint=root / "models" / "uni" / "uni_vit.pt",
    )


def _validate_paths(sp: StainingPaths) -> None:
    """Raise FileNotFoundError if any expected path is missing on disk."""
    for field_name in ("banff_config", "reinhard_config", "binarization_model", "uni_checkpoint"):
        p = getattr(sp, field_name)
        if not p.exists():
            raise FileNotFoundError(f"Expected path for staining '{sp.staining}' not found: {p}")


def staining_paths(
    staining: Optional[str] = None,
    root: Path = DATA_ROOT,
) -> Union[StainingPaths, dict]:
    """Return configured asset paths for one staining or all stainings.

    When called with no arguments, returns a ``dict[str, StainingPaths]``
    mapping every key in STAININGS to its resolved paths, and validates
    that all expected paths exist on disk (raises FileNotFoundError for the
    first missing path).

    A staining name returns a single ``StainingPaths`` without checking files.

    Parameters
    ----------
    staining
        One of :data:`STAININGS`, or ``None`` to return all stain types.
    root
        Root path. Defaults to :data:`DATA_ROOT` (or
        ``BANFFNET_DATA_ROOT`` env var).
    """
    if staining is None:
        result: dict = {}
        for s in STAININGS:
            sp = _make_staining_paths(s, root)
            _validate_paths(sp)
            result[s] = sp
        return result
    s = staining.upper()
    if s not in STAININGS:
        raise ValueError(f"Unknown staining {staining!r}; expected one of {STAININGS}")
    return _make_staining_paths(s, root)
