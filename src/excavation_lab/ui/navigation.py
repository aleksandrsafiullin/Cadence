"""Path and title keys for the seven pages.

The page server draws this order. There is no Streamlit navigation.
"""

from __future__ import annotations

# page id, URL path, catalog attribute
PAGE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("video", "/", "page_video"),
    ("data", "/data", "page_data"),
    ("training", "/training", "page_training"),
    ("analysis", "/analysis", "page_analysis"),
    ("quality", "/quality", "page_quality"),
    ("actions", "/actions", "page_actions"),
    ("events", "/events", "page_events"),
)
