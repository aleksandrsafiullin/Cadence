"""Per-camera zones in normalized coordinates.

A polygon is a camera setting, not a detector class, and it does not encode
which side of the frame is the face. ``x * frame_width`` is applied when a
frame is used, from that frame's width. A view change marks the previous
polygons ``invalid_for_view``; those vertices are not a working area.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import yaml
from pydantic import ValidationError

from excavation_lab.errors import LabError
from excavation_lab.schemas import ZONE_NAMES, ViewConfig, configs_dir

ZONE_VALID = "valid"
ZONE_INVALID_FOR_VIEW = "invalid_for_view"
_ZONE_FIELDS = frozenset(ZONE_NAMES)
_VERTEX_FIELDS = frozenset(("x", "y"))


@dataclass(frozen=True, slots=True)
class CameraZones:
    """One ``view_id``. Empty polygons are allowed. Null identity is unknown."""

    view_id: str
    site_id: str | None
    camera_id: str | None
    equipment_family: str | None
    working_area: tuple[tuple[float, float], ...]
    dig_zone: tuple[tuple[float, float], ...]
    loading_zone: tuple[tuple[float, float], ...]
    entry_zone: tuple[tuple[float, float], ...]
    exit_zone: tuple[tuple[float, float], ...]
    status: str = ZONE_VALID
    measured: bool = False

    def polygon(self, name: str) -> tuple[tuple[float, float], ...]:
        if name not in _ZONE_FIELDS:
            raise LabError("unknown_zone", f"Неизвестная зона: {name}")
        vertices = getattr(self, name)
        return tuple(vertices)

    def invalidated(self) -> CameraZones:
        """Same vertices, refused for association after the camera moves."""
        return replace(self, status=ZONE_INVALID_FOR_VIEW)


@dataclass
class ZoneBinding:
    """Zones for the current view. A different ``view_id`` drops them."""

    current: CameraZones | None = None
    dropped: CameraZones | None = None

    def bind_view(self, view_id: str | None) -> CameraZones | None:
        """Return the zones association may use for ``view_id``.

        A change stores the previous polygons as ``invalid_for_view`` and
        returns none. The caller does not pass that object in.
        """
        current = self.current
        if current is None:
            return None
        if view_id == current.view_id and current.status == ZONE_VALID:
            return current
        self.dropped = current.invalidated()
        self.current = None
        return None


def views_directory() -> Path:
    return configs_dir() / "views"


def load_view_zones(view_id: str, *, directory: Path | None = None) -> CameraZones:
    """Load ``<directory>/<view_id>.yaml``. The default directory is ``configs/views``."""
    token = _view_token(view_id)
    path = (views_directory() if directory is None else Path(directory)) / f"{token}.yaml"
    zones = load_view_zones_file(path)
    if zones.view_id != token:
        raise LabError(
            "view_mismatch",
            f"Идентификатор в файле ({zones.view_id}) не совпадает с {token}.",
        )
    return zones


def load_view_zones_file(path: Path) -> CameraZones:
    """Load one view file. Polygons are empty or valid. Identity null stays null."""
    if not path.is_file():
        raise LabError("config_missing", f"Нет файла ракурса: {path.name}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LabError("config_invalid", f"Не удалось прочитать YAML: {path.name}") from exc
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise LabError("config_invalid", f"Файл ракурса должен быть YAML-объектом: {path.name}")
    try:
        view = ViewConfig.model_validate(loaded)
    except ValidationError as exc:
        _raise_view_error(exc)
    return _from_config(view)


def polygon_pixels(
    vertices: Sequence[tuple[float, float]],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[float, float], ...]:
    """Scale normalized vertices by this frame. The width is not cached."""
    width = _frame_extent(frame_width)
    height = _frame_extent(frame_height)
    return tuple((float(x) * width, float(y) * height) for x, y in vertices)


def active_working_area_pixels(
    zones: CameraZones | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[float, float], ...] | None:
    """Pixel ``working_area``, or none when the view is missing, empty, or stale."""
    if zones is None or zones.status != ZONE_VALID:
        return None
    if len(zones.working_area) < 3:
        return None
    return polygon_pixels(
        zones.working_area,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def annotation_polygons(
    zones: CameraZones | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Pixel polygons for a drawer. An invalid view contributes nothing."""
    if zones is None or zones.status != ZONE_VALID:
        return ()
    drawn: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for name in ZONE_NAMES:
        vertices = zones.polygon(name)
        if len(vertices) < 3:
            continue
        drawn.append(
            (
                name,
                polygon_pixels(vertices, frame_width=frame_width, frame_height=frame_height),
            )
        )
    return tuple(drawn)


def point_in_polygon(
    x: float,
    y: float,
    vertices: Sequence[tuple[float, float]],
) -> bool:
    """Even-odd test. The boundary counts as inside. No half-frame shortcut."""
    if len(vertices) < 3:
        return False
    if _on_boundary(x, y, vertices):
        return True
    inside = False
    previous_x, previous_y = vertices[-1]
    for current_x, current_y in vertices:
        if (current_y > y) != (previous_y > y):
            cross_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y)
            cross_x += current_x
            if x < cross_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _from_config(view: ViewConfig) -> CameraZones:
    polygons = {name: _vertices(view.zone(name)) for name in ZONE_NAMES}
    return CameraZones(
        view_id=view.view_id,
        site_id=view.site_id,
        camera_id=view.camera_id,
        equipment_family=view.equipment_family,
        measured=view.measured,
        status=ZONE_VALID,
        **polygons,
    )


def _vertices(points: Sequence[Any]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point.x), float(point.y)) for point in points)


def _view_token(view_id: str) -> str:
    if (
        not isinstance(view_id, str)
        or not view_id.strip()
        or view_id in {".", ".."}
        or "/" in view_id
        or "\\" in view_id
        or "\x00" in view_id
    ):
        raise LabError("invalid_view", "Недопустимый идентификатор ракурса.")
    return view_id


def _frame_extent(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabError(
            "invalid_frame",
            "Ширина и высота кадра должны быть положительными целыми.",
        )
    return value


def _on_boundary(x: float, y: float, vertices: Sequence[tuple[float, float]]) -> bool:
    count = len(vertices)
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % count]
        if _on_segment(x, y, x1, y1, x2, y2):
            return True
    return False


def _on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -1e-9:
        return False
    length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return dot <= length_sq + 1e-9


def _raise_view_error(exc: ValidationError) -> NoReturn:
    messages = " ".join(str(error.get("msg", "")) for error in exc.errors())
    locations = [error.get("loc", ()) for error in exc.errors()]
    about_zone = any(
        any(part in _ZONE_FIELDS or part in _VERTEX_FIELDS for part in loc) for loc in locations
    )
    if "at least 3" in messages:
        text = "У зоны меньше трёх вершин."
    elif "zero area" in messages:
        text = "Площадь зоны равна нулю."
    elif about_zone:
        text = "Координата зоны должна быть в диапазоне от 0 до 1."
    else:
        raise LabError("config_invalid", "Файл ракурса не соответствует схеме.") from exc
    raise LabError("invalid_zone", text) from exc
