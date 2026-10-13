"""Status stripe tone. Unknown stays neutral and is not a zero."""

from __future__ import annotations

import re

import pytest

from excavation_lab.ui.status import status_markup, status_tone

_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")


@pytest.mark.parametrize(
    ("status", "tone"),
    [
        ("queued", "accent"),
        ("running", "accent"),
        ("completed", "success"),
        ("failed", "danger"),
        ("cancel_requested", "warning"),
        ("cancelled", "neutral"),
        (None, "neutral"),
        ("stopping", "neutral"),
    ],
)
def test_status_tone(status: str | None, tone: str) -> None:
    assert status_tone(status) == tone


def test_render_status_escapes_text_and_rejects_an_unknown_tone() -> None:
    danger = status_markup('кадр <b> & "x"', "danger")
    unknown = status_markup("не рассчитано", "stopping")
    captured = [danger, unknown]
    assert danger == status_markup('кадр <b> & "x"', "danger")
    assert unknown == status_markup("не рассчитано", "stopping")
    assert 'class="lab-alert"' in danger
    assert 'data-tone="danger"' in danger
    assert 'class="lab-status-mark"' in danger
    assert "кадр &lt;b&gt; &amp; &quot;x&quot;" in danger
    assert "style=" not in danger
    assert 'data-tone="neutral"' in unknown
    assert "не рассчитано" in unknown
    assert "style=" not in unknown
    for body in captured:
        assert _ZERO_TOKEN.search(body) is None
