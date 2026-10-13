"""Job-status stripe. Callers keep the Russian sentence."""

from __future__ import annotations

import html

_TONES = frozenset({"accent", "success", "warning", "danger", "neutral"})
_TONE_BY_STATUS = {
    "queued": "accent",
    "running": "accent",
    "completed": "success",
    "failed": "danger",
    "cancel_requested": "warning",
    "cancelled": "neutral",
}


def status_tone(status: str | None) -> str:
    """Stripe tone for one job status. Anything outside the six is neutral."""
    if status is None:
        return "neutral"
    return _TONE_BY_STATUS.get(status, "neutral")


def status_markup(text: str, tone: str) -> str:
    """Escaped ``.lab-alert`` row. A tone outside the five is neutral."""
    chosen = tone if tone in _TONES else "neutral"
    return (
        '<div class="lab-alert" data-tone="'
        + chosen
        + '"><span class="lab-status-mark"></span>'
        + html.escape(text)
        + "</div>"
    )

