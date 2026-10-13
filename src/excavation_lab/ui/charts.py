"""This chart sets no named theme, which is how this app applies the design rule against plotly, plotly_dark, and ggplot2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import altair as alt

from excavation_lab.ui.tokens import theme_custom_properties

_FONT_SIZE = 12
_LINE_WIDTH = 1
_PADDING = {"left": 56, "right": 16, "bottom": 48, "top": 16}
_CHROME = (
    "--pdm-chart-bg",
    "--pdm-font",
    "--pdm-chart-text",
    "--pdm-chart-axis",
    "--pdm-chart-grid",
    "--pdm-surface",
    "--pdm-border",
    "--pdm-text",
)


def loss_chart(
    epochs: Sequence[int],
    losses: Sequence[float],
    theme: str,
) -> alt.Chart:
    """Measured training loss as one solid observed line."""
    rows = [
        {"epoch": _epoch(epoch), "loss": _finite(loss)}
        for epoch, loss in zip(epochs, losses, strict=True)
    ]
    return _styled(theme, "--pdm-series-observed", lambda colors: _loss_mark(rows, colors))


def ap50_chart(rows: Sequence[Mapping[str, object]], theme: str) -> alt.Chart:
    """AP50 for one measurement. Every bar uses the accent token.

    Both locales keep the field name ``класс`` as the axis text. The
    encoding is not renamed, and this chart sets no chart-level title.
    """
    records = [_ap50_row(row) for row in rows]
    return _styled(theme, "--pdm-accent", lambda colors: _ap50_mark(records, colors))


def _styled(
    theme: str,
    role: str,
    build: Callable[[Mapping[str, str]], alt.Chart],
) -> alt.Chart:
    colors = _require(theme, role, *_CHROME)
    with alt.theme.enable("none"):
        return build(colors)


def _require(theme: str, *names: str) -> dict[str, str]:
    props = theme_custom_properties(theme)
    for name in names:
        value = props.get(name)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(name)
    return props


def _loss_mark(rows: Sequence[Mapping[str, float]], colors: Mapping[str, str]) -> alt.Chart:
    observed = colors["--pdm-series-observed"]
    chart = (
        alt.Chart(alt.Data(values=list(rows)))
        .mark_line(color=observed, strokeWidth=2, interpolate="linear")
        .encode(
            x=alt.X("epoch:Q", axis=_axis(colors, grid=False, line=colors["--pdm-chart-axis"])),
            y=alt.Y("loss:Q", axis=_axis(colors, grid=True, line=colors["--pdm-chart-grid"])),
            tooltip=[alt.Tooltip("epoch:Q"), alt.Tooltip("loss:Q")],
        )
    )
    return _chrome(chart, colors)


def _ap50_mark(rows: Sequence[Mapping[str, object]], colors: Mapping[str, str]) -> alt.Chart:
    accent = colors["--pdm-accent"]
    chart = (
        alt.Chart(alt.Data(values=list(rows)))
        .mark_bar(color=accent)
        .encode(
            x=alt.X(
                "класс:N",
                sort=None,
                axis=_axis(colors, grid=False, line=colors["--pdm-chart-axis"]),
            ),
            y=alt.Y("AP50:Q", axis=_axis(colors, grid=True, line=colors["--pdm-chart-grid"])),
            tooltip=[alt.Tooltip("класс:N"), alt.Tooltip("AP50:Q")],
        )
    )
    return _chrome(chart, colors)


def _chrome(chart: alt.Chart, colors: Mapping[str, str]) -> alt.Chart:
    font = colors["--pdm-font"]
    text = colors["--pdm-chart-text"]
    paper = colors["--pdm-chart-bg"]
    # Vega-Lite config has no hover chrome. usermeta.hover is what to_dict() keeps.
    hover = {
        "background": colors["--pdm-surface"],
        "border": colors["--pdm-border"],
        "text": colors["--pdm-text"],
        "fontSize": _FONT_SIZE,
    }
    return chart.properties(
        background=paper,
        padding=_PADDING,
        width="container",
        height=300,
        view={"fill": paper, "stroke": None},
        usermeta={"hover": hover},
    ).configure(
        font=font,
        axis=_axis_config(font, text),
        axisX=_axis_side(grid=False, line=colors["--pdm-chart-axis"]),
        axisY=_axis_side(
            grid=True,
            line=colors["--pdm-chart-grid"],
            grid_color=colors["--pdm-chart-grid"],
        ),
        legend=alt.LegendConfig(disable=True),
        view=alt.ViewConfig(fill=paper, stroke=None),
    )


def _axis(colors: Mapping[str, str], *, grid: bool, line: str) -> alt.Axis:
    text = colors["--pdm-chart-text"]
    font = colors["--pdm-font"]
    return alt.Axis(
        grid=grid,
        domain=True,
        domainColor=line,
        domainWidth=_LINE_WIDTH,
        gridColor=colors["--pdm-chart-grid"],
        gridWidth=_LINE_WIDTH,
        tickColor=line,
        tickWidth=_LINE_WIDTH,
        labelColor=text,
        labelFont=font,
        labelFontSize=_FONT_SIZE,
        titleColor=text,
        titleFont=font,
        titleFontSize=_FONT_SIZE,
    )


def _axis_config(font: str, text: str) -> alt.AxisConfig:
    return alt.AxisConfig(
        labelFont=font,
        labelFontSize=_FONT_SIZE,
        labelColor=text,
        titleFont=font,
        titleFontSize=_FONT_SIZE,
        titleColor=text,
        domainWidth=_LINE_WIDTH,
        gridWidth=_LINE_WIDTH,
        tickWidth=_LINE_WIDTH,
    )


def _axis_side(*, grid: bool, line: str, grid_color: str | None = None) -> alt.AxisConfig:
    kwargs: dict[str, Any] = {
        "grid": grid,
        "domain": True,
        "domainColor": line,
        "domainWidth": _LINE_WIDTH,
        "tickColor": line,
        "tickWidth": _LINE_WIDTH,
        "gridWidth": _LINE_WIDTH,
    }
    if grid_color is not None:
        kwargs["gridColor"] = grid_color
    return alt.AxisConfig(**kwargs)


def _ap50_row(row: Mapping[str, object]) -> dict[str, object]:
    label = row["класс"]
    if not isinstance(label, str) or not label:
        raise TypeError("AP50 class label must be a string")
    return {"класс": label, "AP50": _finite(row["AP50"])}


def _epoch(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("epoch must be an int")
    if value < 0:
        raise ValueError("epoch must be zero or positive")
    return value


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("chart value must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("chart value must be finite")
    return number
