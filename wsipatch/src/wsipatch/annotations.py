"""Parse ASAP XML annotations and convert them to level-0 ROI specifications.

``ASAPAnnotations.from_slide`` provides a full-slide fallback. ``asap_to_rois``
and ``load_asap_rois`` support group and type filters and clip negative origins.
"""

from __future__ import annotations

import logging
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

from .types import ROISpec

logger = logging.getLogger(__name__)


class AnnotationParseError(Exception):
    pass


class Annotation:
    """One ASAP annotation with its raw coordinate list."""

    def __init__(
        self,
        name: str,
        annotation_type: str,
        part_of_group: str,
        color: str,
        coordinates: typing.List[typing.Tuple[int, float, float]],
    ):
        self.name = name
        self.type = annotation_type
        self.part_of_group = part_of_group
        self.color = color
        self.coordinates = coordinates

    def __repr__(self) -> str:
        return (
            f"Annotation(Name={self.name}, Type={self.type}, "
            f"PartOfGroup={self.part_of_group}, Color={self.color}, "
            f"NumCoordinates={len(self.coordinates)})"
        )

    def get_enclosing_rectangle(
        self,
        downscaling_factor: float = 1.0,
    ) -> typing.Tuple[int, int, int, int]:
        """Bounding box ``(x_min, x_max, y_min, y_max)`` after optional downscale."""
        import numpy as np

        xs = [x for _, x, _ in self.coordinates]
        ys = [y for _, _, y in self.coordinates]
        return (
            int(np.floor(min(xs) * downscaling_factor)),
            int(np.ceil(max(xs) * downscaling_factor)),
            int(np.floor(min(ys) * downscaling_factor)),
            int(np.ceil(max(ys) * downscaling_factor)),
        )

    def get_region_to_read(
        self,
        downscaling_factor: float = 1.0,
    ) -> typing.Tuple[int, int, int, int]:
        """``(x, y, width, height)`` form of the enclosing rectangle."""
        x_min, x_max, y_min, y_max = self.get_enclosing_rectangle(downscaling_factor)
        return x_min, y_min, x_max - x_min, y_max - y_min

    def get_polygon_points(
        self,
        downscaling_factor: float = 1.0,
    ) -> typing.List[typing.Tuple[int, int]]:
        """Polygon points in ASAP order, optionally downscaled."""
        return [
            (int(x * downscaling_factor), int(y * downscaling_factor))
            for _, x, y in sorted(self.coordinates, key=lambda c: c[0])
        ]


class ASAPAnnotations:
    """Collection of :class:`Annotation` parsed from ASAP XML."""

    def __init__(self, xml_content: str):
        self.annotations: typing.List[Annotation] = []
        self._parse_xml(xml_content)

    def _parse_xml(self, xml_content: str) -> None:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error("Failed to parse XML: %s", e)
            raise AnnotationParseError(f"Invalid XML content: {e}") from e

        anns = root.find("Annotations")
        if anns is None:
            logger.warning("No Annotations element found in XML")
            return
        for elem in anns.findall("Annotation"):
            ann = self._parse_annotation_element(elem)
            if ann is not None:
                self.annotations.append(ann)

    @staticmethod
    def _parse_annotation_element(element: ET.Element) -> typing.Optional[Annotation]:
        try:
            name = element.get("Name", "")
            ann_type = element.get("Type", "")
            group = element.get("PartOfGroup", "")
            color = element.get("Color", "")

            coords_elem = element.find("Coordinates")
            if coords_elem is None:
                logger.warning("No coordinates for annotation %r", name)
                return None

            coords: typing.List[typing.Tuple[int, float, float]] = []
            for c in coords_elem.findall("Coordinate"):
                order = int(c.get("Order", 0))
                x = float(c.get("X", "0").replace(",", "."))
                y = float(c.get("Y", "0").replace(",", "."))
                coords.append((order, x, y))

            if not coords:
                logger.warning("Empty coordinates for annotation %r", name)
                return None
            return Annotation(name, ann_type, group, color, coords)
        except (ValueError, AttributeError) as e:
            logger.error("Failed to parse annotation element: %s", e)
            return None

    def __iter__(self):
        return iter(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> Annotation:
        return self.annotations[index]

    def filter_by_group(self, group: str) -> typing.List[Annotation]:
        return [a for a in self.annotations if a.part_of_group == group]

    def filter_by_type(self, ann_type: str) -> typing.List[Annotation]:
        return [a for a in self.annotations if a.type == ann_type]

    def get_groups(self) -> typing.Set[str]:
        return {a.part_of_group for a in self.annotations}

    def get_types(self) -> typing.Set[str]:
        return {a.type for a in self.annotations}

    def to_dict(self) -> dict:
        return {
            "annotations": [
                {
                    "name": a.name,
                    "type": a.type,
                    "group": a.part_of_group,
                    "color": a.color,
                    "num_coordinates": len(a.coordinates),
                }
                for a in self.annotations
            ],
            "total": len(self.annotations),
            "groups": list(self.get_groups()),
            "types": list(self.get_types()),
        }

    @classmethod
    def from_xml_file(
        cls, file_path: typing.Union[str, Path]
    ) -> typing.Optional["ASAPAnnotations"]:
        p = Path(file_path)
        if not p.exists():
            logger.error("Annotation file not found: %s", p)
            return None
        try:
            return cls(p.read_text(encoding="utf-8"))
        except (IOError, UnicodeDecodeError) as e:
            logger.error("Failed to read annotation file %s: %s", p, e)
            return None
        except AnnotationParseError as e:
            logger.error("Failed to parse annotations from %s: %s", p, e)
            return None

    @classmethod
    def from_slide(
        cls,
        slide_path: typing.Union[str, Path],
        group: str = "tissue",
        annotation_name: str = "full_slide",
    ) -> typing.Optional["ASAPAnnotations"]:
        """Synthetic single-annotation covering the whole slide.

        Used as the fallback when no XML is available.
        """
        try:
            import openslide
        except ImportError:
            logger.error("openslide-python is required for from_slide()")
            return None

        p = Path(slide_path)
        if not p.exists():
            logger.error("Slide file not found: %s", p)
            return None

        try:
            slide = openslide.OpenSlide(str(p))
            w, h = slide.dimensions
            slide.close()
        except openslide.OpenSlideError as e:
            logger.error("Failed to open slide %s: %s", p, e)
            return None

        corners = [
            (0, 0.0, 0.0),
            (1, float(w), 0.0),
            (2, float(w), float(h)),
            (3, 0.0, float(h)),
        ]
        ann = Annotation(annotation_name, "Polygon", group, "#73d216", corners)
        inst = cls.__new__(cls)
        inst.annotations = [ann]
        return inst


def asap_to_rois(
    asap_annotations: typing.Iterable,
    *,
    groups: typing.Optional[typing.Sequence[str]] = None,
    types: typing.Optional[typing.Sequence[str]] = None,
    downscaling_factor: float = 1.0,
    min_area: int = 1,
    clip_to_origin: bool = True,
) -> typing.List[ROISpec]:
    """Convert an iterable of ``Annotation``-likes into :class:`ROISpec` s.

    Parameters
    ----------
    asap_annotations
        Anything iterating ``Annotation``-like values (an
        :class:`ASAPAnnotations`, ``asap.filter_by_group(...)``, a list...).
    groups, types
        Optional whitelists on ``part_of_group`` / ``type``.
    downscaling_factor
        Forwarded to :meth:`Annotation.get_region_to_read`. ``1.0`` keeps
        level-0 coordinates (the wsipatch convention).
    min_area
        Drop annotations whose bbox has fewer than this many level-0 pixels.
    clip_to_origin
        Clamp negative ``x``/``y`` to 0 (and shrink ``w``/``h`` accordingly).
        :class:`ROISpec` requires non-negative origins.
    """
    out: typing.List[ROISpec] = []
    for ann in asap_annotations:
        ann_group = getattr(ann, "part_of_group", "")
        ann_type = getattr(ann, "type", "")
        # Groups may contain several ROIs; only Name provides ROI identity.
        ann_name = (getattr(ann, "name", "") or "").strip() or None

        if groups is not None and ann_group not in groups:
            continue
        if types is not None and ann_type not in types:
            continue

        try:
            x, y, w, h = ann.get_region_to_read(downscaling_factor)
        except Exception as e:
            logger.warning(
                "Skipping annotation %r: get_region_to_read failed: %s",
                ann_name,
                e,
            )
            continue
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        if clip_to_origin:
            if x < 0:
                w += x
                x = 0
            if y < 0:
                h += y
                y = 0
        if w < 0 or h < 0 or w * h < min_area:
            logger.debug(
                "Skipping annotation %r: degenerate bbox (w=%d, h=%d)",
                ann_name,
                w,
                h,
            )
            continue

        out.append(ROISpec(x0=x, y0=y, w=w, h=h, name=ann_name))
    return out


def load_asap_rois(
    path: typing.Union[str, Path],
    *,
    groups: typing.Optional[typing.Sequence[str]] = None,
    types: typing.Optional[typing.Sequence[str]] = None,
    clip_to_origin: bool = True,
    min_area: int = 1,
) -> typing.List[ROISpec]:
    """Parse an ASAP XML file and return its annotations as :class:`ROISpec` s."""
    p = Path(path)
    if not p.exists():
        logger.warning("ASAP XML not found: %s", p)
        return []
    asap = ASAPAnnotations.from_xml_file(p)
    if asap is None or len(asap) == 0:
        return []
    return asap_to_rois(
        asap, groups=groups, types=types, clip_to_origin=clip_to_origin, min_area=min_area
    )
