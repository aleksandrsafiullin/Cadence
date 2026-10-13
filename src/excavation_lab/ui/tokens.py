"""Design tokens from the package stylesheets.

This module does not import Streamlit. Charts read colors from
``theme_custom_properties``.
"""

from __future__ import annotations

import re
from pathlib import Path

LAB_THEME_CSS = Path(__file__).with_name("lab_theme.css")

_LIGHT_MARKER = '[data-theme="light"]'
_CUSTOM_PROPERTY = re.compile(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);")


def combined_css() -> str:
    """Portable stylesheet."""
    return LAB_THEME_CSS.read_text(encoding="utf-8")


def theme_custom_properties(theme: str) -> dict[str, str]:
    """Custom properties for ``dark`` or ``light``.

    Dark values are the declarations before ``[data-theme="light"]``.
    Light values override that map. Any other theme name raises.
    """
    if theme == "dark":
        which = "dark"
    elif theme == "light":
        which = "light"
    else:
        raise ValueError(f"unknown theme: {theme!r}")
    portable = LAB_THEME_CSS.read_text(encoding="utf-8")
    dark_block, light_block = _split_portable(portable)
    dark = _properties(dark_block)
    if which == "dark":
        return dark
    merged = dict(dark)
    merged.update(_properties(light_block))
    return merged


def _split_portable(portable: str) -> tuple[str, str]:
    parts = portable.split(_LIGHT_MARKER, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("portable stylesheet has no light theme block")
    return parts[0], parts[1]


def _properties(block: str) -> dict[str, str]:
    return {name: value.strip() for name, value in _CUSTOM_PROPERTY.findall(block)}
