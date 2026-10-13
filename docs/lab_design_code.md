# Lab Design Code

Переносимый визуальный код Predictive Maintenance Lab. Источник правды: `src/pdm/ui_theme.py` и `src/pdm/visualization/explorer.css`.

Подключай этот файл в другом ML-приложении как есть. Имена токенов `--pdm-*` не переименовывай: так обе программы остаются одной темой.

`docs/wow_page_design.md` — другая палитра (cyan / amber / coral). Её не смешивать с этим кодом.

## Характер

Тихий приборный интерфейс. Нейтральный фон, одна синяя акцентная краска, плоские поверхности, граница 1px. Тень есть только в светлой теме. Статус не закрашивает карточку: он живёт в иконке и в полоске 3px слева.

Тёмная тема — основная. Светлая — та же структура, другие значения токенов. Переключатель: `Light` | `Dark`. Выбор сохраняется между экранами и после перезагрузки.

Хром не градиентный. Декоративных графиков, выдуманных точек и «живых» анимаций данных нет. Цвет на графике означает роль ряда, а не украшение.

## Как подключить

На корневой элемент:

```html
<html data-theme="dark">
```

Дальше вставь блок «Токены и базовые компоненты» одним файлом стилей. Компоненты используют только `var(--pdm-…)`. Hex в компонентах не писать.

Смена темы — смена `data-theme` на `light` или `dark`. Оба набора токенов уже в файле.

## Токены и базовые компоненты

```css
:root, [data-theme="dark"] {
  color-scheme: dark;
  --pdm-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Inter", Roboto, "Helvetica Neue", Arial, sans-serif;
  --pdm-bg: #0F0F11;
  --pdm-sidebar: #161618;
  --pdm-surface: #1C1C1E;
  --pdm-surface-subtle: #232326;
  --pdm-input: #1C1C1E;
  --pdm-control: #2C2C2E;
  --pdm-control-hover: #3A3A3C;
  --pdm-text: #F2F2F4;
  --pdm-text-secondary: #D1D1D6;
  --pdm-muted: #98989D;
  --pdm-border: #38383A;
  --pdm-border-soft: #2C2C2E;
  --pdm-border-strong: #48484A;
  --pdm-accent: #0A84FF;
  --pdm-accent-hover: #409CFF;
  --pdm-accent-ink: #000000;
  --pdm-accent-soft: #10263D;
  --pdm-accent-text: #64B5FF;
  --pdm-success: #30D158;
  --pdm-warning: #FF9F0A;
  --pdm-danger: #FF453A;
  --pdm-focus-ring: rgba(10, 132, 255, 0.35);
  --pdm-chart-bg: #1C1C1E;
  --pdm-chart-text: #98989D;
  --pdm-chart-grid: #2C2C2E;
  --pdm-chart-axis: #38383A;
  --pdm-series-observed: #A1A1A6;
  --pdm-series-future-actual: rgba(115, 115, 120, 0.5);
  --pdm-series-forecast: #0A84FF;
  --pdm-series-band: rgba(10, 132, 255, 0.16);
  --pdm-series-band-line: rgba(10, 132, 255, 0.45);
  --pdm-series-reference: #F2F2F4;
  --pdm-zone-green: #30D158;
  --pdm-zone-yellow: #FFD60A;
  --pdm-zone-red: #FF453A;
  --pdm-zone-unknown: #7C7C80;
  --pdm-zone-yellow-fill: rgba(255, 214, 10, 0.08);
  --pdm-zone-red-fill: rgba(255, 69, 58, 0.09);
  --pdm-shadow: none;
  --pdm-shadow-overlay: none;
  --pdm-series-1: #0A84FF;
  --pdm-series-2: #A1A1A6;
  --pdm-series-3: #FF9F0A;
  --pdm-series-4: #30D158;
  --pdm-series-5: #BF5AF2;
  --pdm-series-6: #FF453A;
}

[data-theme="light"] {
  color-scheme: light;
  --pdm-bg: #F5F5F7;
  --pdm-sidebar: #EDEDF0;
  --pdm-surface: #FFFFFF;
  --pdm-surface-subtle: #FAFAFC;
  --pdm-input: #FFFFFF;
  --pdm-control: #FFFFFF;
  --pdm-control-hover: #F0F0F3;
  --pdm-text: #1D1D1F;
  --pdm-text-secondary: #424245;
  --pdm-muted: #636366;
  --pdm-border: #D2D2D7;
  --pdm-border-soft: #E5E5EA;
  --pdm-border-strong: #C7C7CC;
  --pdm-accent: #0071E3;
  --pdm-accent-hover: #0062C4;
  --pdm-accent-ink: #FFFFFF;
  --pdm-accent-soft: #E8F1FC;
  --pdm-accent-text: #0058B0;
  --pdm-success: #1E7B34;
  --pdm-warning: #B35C00;
  --pdm-danger: #D70015;
  --pdm-focus-ring: rgba(0, 113, 227, 0.28);
  --pdm-chart-bg: #FFFFFF;
  --pdm-chart-text: #636366;
  --pdm-chart-grid: #E5E5EA;
  --pdm-chart-axis: #D2D2D7;
  --pdm-series-observed: #6E6E73;
  --pdm-series-future-actual: rgba(142, 142, 147, 0.5);
  --pdm-series-forecast: #0071E3;
  --pdm-series-band: rgba(0, 113, 227, 0.10);
  --pdm-series-band-line: rgba(0, 113, 227, 0.45);
  --pdm-series-reference: #1D1D1F;
  --pdm-zone-green: #1E7B34;
  --pdm-zone-yellow: #C28800;
  --pdm-zone-red: #D70015;
  --pdm-zone-unknown: #AEAEB2;
  --pdm-zone-yellow-fill: rgba(194, 136, 0, 0.08);
  --pdm-zone-red-fill: rgba(215, 0, 21, 0.06);
  --pdm-shadow: 0 1px 2px color-mix(in srgb, var(--pdm-text) 4%, transparent);
  --pdm-shadow-overlay: 0 4px 16px color-mix(in srgb, var(--pdm-text) 8%, transparent);
  --pdm-series-1: #0071E3;
  --pdm-series-2: #6E6E73;
  --pdm-series-3: #B35C00;
  --pdm-series-4: #1E7B34;
  --pdm-series-5: #8944AB;
  --pdm-series-6: #D70015;
}

body {
  margin: 0;
  background: var(--pdm-bg);
  color: var(--pdm-text);
  font-family: var(--pdm-font);
  font-size: 14px;
  line-height: 21px;
  font-weight: 400;
}

.lab-app { display: flex; min-height: 100vh; background: var(--pdm-bg); color: var(--pdm-text); }
.lab-sidebar {
  width: 21rem; flex: 0 0 21rem; box-sizing: border-box;
  background: var(--pdm-sidebar); color: var(--pdm-text);
  border-right: 1px solid var(--pdm-border-soft); padding: 24px 16px;
}
.lab-main { flex: 1 1 auto; min-width: 0; max-width: 1600px; padding: 32px 48px 64px; }
.lab-main.lab-wide { max-width: 1900px; }

.lab-kicker {
  margin: 16px 12px 6px; font-size: 11px; line-height: 14px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--pdm-muted);
}
.lab-title { margin: 0; font-size: 28px; line-height: 34px; font-weight: 600; letter-spacing: -0.02em; color: var(--pdm-text); }
.lab-section { margin: 0; font-size: 17px; line-height: 24px; font-weight: 600; letter-spacing: -0.01em; color: var(--pdm-text); }
.lab-lead { max-width: 720px; margin: 8px 0 16px; font-size: 15px; line-height: 22px; font-weight: 400; color: var(--pdm-text-secondary); }
.lab-label { font-size: 13px; line-height: 18px; font-weight: 500; color: var(--pdm-text); }
.lab-caption { font-size: 12px; line-height: 17px; font-weight: 400; color: var(--pdm-muted); }
.lab-link { color: var(--pdm-accent-text); text-decoration: none; }
.lab-link:hover { text-decoration: underline; }

.lab-btn, .lab-input, .lab-select {
  box-sizing: border-box; min-height: 36px; border-radius: 8px;
  font-family: inherit; font-size: 14px; line-height: 20px; font-weight: 500;
  transition: border-color .15s ease-out, background .15s ease-out, color .15s ease-out;
}
.lab-btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 0 16px; background: var(--pdm-control); color: var(--pdm-text);
  border: 1px solid var(--pdm-border); cursor: pointer;
}
.lab-btn:hover:not(:disabled) { background: var(--pdm-control-hover); border-color: var(--pdm-border-strong); }
.lab-btn:disabled { opacity: .4; cursor: not-allowed; }
.lab-btn-primary { background: var(--pdm-accent); border-color: var(--pdm-accent); color: var(--pdm-accent-ink); font-weight: 600; }
.lab-btn-primary:hover:not(:disabled) { background: var(--pdm-accent-hover); border-color: var(--pdm-accent-hover); color: var(--pdm-accent-ink); }
.lab-btn-quiet { background: transparent; border-color: transparent; color: var(--pdm-accent-text); min-height: 28px; padding: 0 8px; font-size: 13px; }
.lab-nav-btn {
  display: flex; width: 100%; justify-content: flex-start; text-align: left;
  min-height: 36px; padding: 0 12px; border: 0; border-radius: 8px;
  background: transparent; color: var(--pdm-text); font-weight: 500; cursor: pointer;
}
.lab-nav-btn:hover:not(:disabled) { background: var(--pdm-control-hover); }
.lab-nav-btn[aria-current="page"] { background: var(--pdm-accent-soft); color: var(--pdm-accent-text); font-weight: 600; }
.lab-nav-btn:disabled { opacity: 1; color: var(--pdm-muted); cursor: not-allowed; }

.lab-input, .lab-select {
  width: 100%; padding: 0 12px; background: var(--pdm-input); color: var(--pdm-text);
  border: 1px solid var(--pdm-border); font-weight: 400;
}
.lab-input::placeholder { color: var(--pdm-muted); }
.lab-input:hover, .lab-select:hover { border-color: var(--pdm-border-strong); }
.lab-input:focus, .lab-select:focus, .lab-btn:focus-visible, .lab-nav-btn:focus-visible {
  outline: 2px solid var(--pdm-accent); outline-offset: 2px;
}
.lab-input:focus, .lab-select:focus { border-color: var(--pdm-accent); box-shadow: 0 0 0 3px var(--pdm-focus-ring); outline: none; }
.lab-input:disabled, .lab-select:disabled { opacity: .4; cursor: not-allowed; }

.lab-segmented { display: inline-flex; }
.lab-segmented button {
  min-height: 32px; padding: 0 12px; background: var(--pdm-control); color: var(--pdm-text);
  border: 1px solid var(--pdm-border); font-size: 13px; line-height: 18px; font-weight: 500; cursor: pointer;
}
.lab-segmented button + button { margin-left: -1px; }
.lab-segmented button:first-child { border-radius: 8px 0 0 8px; }
.lab-segmented button:last-child { border-radius: 0 8px 8px 0; }
.lab-segmented button[aria-checked="true"] { background: var(--pdm-accent-soft); color: var(--pdm-accent-text); font-weight: 600; }

.lab-card, .lab-metric, .lab-table-wrap, .lab-chart, .lab-alert, .lab-empty, .lab-expander {
  background: var(--pdm-surface); border: 1px solid var(--pdm-border); border-radius: 12px;
  box-shadow: var(--pdm-shadow);
}
.lab-card { padding: 24px; }
.lab-card-head {
  background: var(--pdm-control); border-bottom: 1px solid var(--pdm-border);
  border-radius: 11px 11px 0 0; padding: 10px 12px; text-align: center;
  font-size: 13px; line-height: 18px; font-weight: 600;
}
.lab-card-body { padding: 12px 14px; }
.lab-card-foot { margin-top: auto; padding: 10px 14px 0; border-top: 1px solid var(--pdm-border); }
.lab-metric { padding: 16px; }
.lab-metric b { display: block; font-size: 28px; line-height: 32px; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.lab-metric span { display: block; margin-bottom: 4px; color: var(--pdm-muted); font-size: 12px; line-height: 16px; font-weight: 500; }
.lab-empty { padding: 24px; }
.lab-empty p { margin: 8px 0 0; color: var(--pdm-text-secondary); }
.lab-row + .lab-row { border-top: 1px solid var(--pdm-border-soft); padding-top: 12px; margin-top: 12px; }

.lab-table { width: 100%; border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
.lab-table th {
  position: sticky; top: 0; background: var(--pdm-surface-subtle); color: var(--pdm-muted);
  font-size: 12px; font-weight: 600; text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--pdm-border-strong);
}
.lab-table td { padding: 8px 12px; font-size: 13px; border-bottom: 1px solid var(--pdm-border-soft); }
.lab-table tr:last-child td { border-bottom: 0; }
.lab-table-wrap { overflow: hidden; }

.lab-alert {
  background: var(--pdm-surface-subtle); border: 1px solid var(--pdm-border-soft);
  border-left: 3px solid var(--pdm-accent); border-radius: 12px; padding: 12px 16px;
  box-shadow: none;
}
.lab-alert[data-tone="success"] { border-left-color: var(--pdm-success); }
.lab-alert[data-tone="warning"] { border-left-color: var(--pdm-warning); }
.lab-alert[data-tone="danger"] { border-left-color: var(--pdm-danger); }
.lab-alert svg { color: var(--pdm-accent); }
.lab-alert[data-tone="success"] svg { color: var(--pdm-success); }
.lab-alert[data-tone="warning"] svg { color: var(--pdm-warning); }
.lab-alert[data-tone="danger"] svg { color: var(--pdm-danger); }

.lab-tabs { display: flex; gap: 16px; border-bottom: 1px solid var(--pdm-border-soft); }
.lab-tabs a, .lab-tabs button {
  background: none; border: 0; border-bottom: 2px solid transparent; margin-bottom: -1px;
  padding: 8px 0; color: var(--pdm-muted); font-size: 14px; line-height: 20px; font-weight: 500; cursor: pointer;
}
.lab-tabs [aria-selected="true"] { color: var(--pdm-text); font-weight: 600; border-bottom-color: var(--pdm-accent); }
.lab-panel { padding-top: 24px; }

.lab-progress { height: 8px; background: var(--pdm-border-soft); border-radius: 999px; overflow: hidden; }
.lab-progress > span { display: block; height: 100%; background: var(--pdm-accent); border-radius: 999px; }
.lab-drop {
  background: var(--pdm-surface-subtle); border: 1px dashed var(--pdm-border-strong);
  border-radius: 12px; padding: 24px; color: var(--pdm-muted); text-align: center;
}
.lab-drop:hover { border-color: var(--pdm-accent); }
.lab-code, code {
  background: var(--pdm-surface-subtle); color: var(--pdm-text-secondary);
  border: 1px solid var(--pdm-border-soft); border-radius: 4px; padding: 0 4px;
  font-size: 0.92em;
}
pre.lab-code { display: block; border-radius: 8px; padding: 12px 16px; overflow: auto; }
.lab-tooltip, .lab-menu {
  background: var(--pdm-surface); color: var(--pdm-text); border: 1px solid var(--pdm-border);
  border-radius: 8px; box-shadow: var(--pdm-shadow-overlay);
}
.lab-tooltip { padding: 8px 12px; max-width: 320px; font-size: 13px; line-height: 18px; }
.lab-menu { padding: 4px; }
.lab-menu button { display: block; width: 100%; text-align: left; min-height: 32px; padding: 0 8px; border: 0; border-radius: 8px; background: transparent; color: var(--pdm-text); }
.lab-menu button:hover, .lab-menu button[data-focused="true"] { background: var(--pdm-control-hover); }
.lab-menu button[aria-selected="true"] { background: var(--pdm-accent-soft); color: var(--pdm-accent-text); }

.lab-expander { background: var(--pdm-surface-subtle); }
.lab-expander summary { padding: 12px 16px; font-size: 14px; line-height: 20px; font-weight: 500; cursor: pointer; }
.lab-expander summary:hover { background: var(--pdm-control-hover); }
.lab-expander .lab-expander-body { padding: 0 16px 16px; border-top: 1px solid var(--pdm-border-soft); }

@media (max-width: 640px) {
  .lab-sidebar { width: 100%; flex-basis: auto; border-right: 0; border-bottom: 1px solid var(--pdm-border-soft); }
  .lab-app { flex-direction: column; }
  .lab-main, .lab-main.lab-wide { padding: 24px 16px 32px; }
}
```

Тепловая шкала, её в CSS-переменную не завернуть:

| Тема | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| dark | `#1C1C1E` | `#1F4E80` | `#0A84FF` |
| light | `#FFFFFF` | `#9CC3F0` | `#0071E3` |

## Типографика

Один стек: `--pdm-font`. Веса только 400, 500, 600. Курсив не используется. Числа в метриках и таблицах: `font-variant-numeric: tabular-nums`.

| Роль | Размер / линия | Вес | Трекинг | Цвет |
| --- | --- | --- | --- | --- |
| Заголовок страницы | 28 / 34 | 600 | −0.02em | text |
| Заголовок секции | 17 / 24 | 600 | −0.01em | text |
| Описание страницы | 15 / 22 | 400 | 0 | text-secondary, max 720px |
| Основной текст | 14 / 21 | 400 | 0 | text |
| Кнопка, пункт меню | 14 / 20 | 500, primary 600 | 0 | text или accent-ink |
| Подпись поля | 13 / 18 | 500 | 0 | text |
| Шапка карточки, сегмент | 13 / 18 | 600 / 500 | 0 | text |
| Подпись, ось, легенда | 12 / 17 | 400 | 0 | muted или chart-text |
| Метка сайдбара | 11 / 14 | 600 | +0.06em, uppercase | muted |
| Значение метрики | 28 / 32 | 600 | −0.02em | text |

Заголовок страницы — одна строка. Под ним ровно одно предложение: что на этом экране делают. Без подзаголовка-слогана.

## Отступы, радиус, движение

Шаг, который уже используется: 4, 8, 12, 16, 24, 32, 48, 64. Между секциями 24. Внутри карточки 24, у плотной карточки тело 12–14. Поля страницы: 32 сверху, 48 по бокам, 64 снизу. Уже 640px: 24 / 16 / 32.

| Радиус | Где |
| --- | --- |
| 4 | inline code |
| 8 | кнопки, поля, навигация, меню, тултип, код |
| 12 | карточка, метрика, таблица, график, алерт, пустое состояние, экспандер, дропзона |
| 999 | прогресс |

Анимация только `150ms ease-out` на цвете, фоне и границе. Без пружин и без анимации самих данных.

Фокус: контур 2px `--pdm-accent`, offset 2px. Поле ввода вместо контура получает границу accent и кольцо `--pdm-focus-ring` на 3px.

Светлая тема: карточки `var(--pdm-shadow)`, всплывающие слои `var(--pdm-shadow-overlay)`. Тёмная тема: тени нет, край держит граница.

## Каркас экрана

Слева узкая колонка. В лаборатории ширина не задана, берётся сайдбар Streamlit; для переноса ставь `21rem`. Фон `--pdm-sidebar`, правая граница `--pdm-border-soft`, padding 24×16. Группы подписями: `PROJECT`, `WORKFLOW`, `APPEARANCE`. Пункт текущей страницы — мягкая заливка accent и текст accent-text, без капсулы на весь сайдбар. Недоступный пункт остаётся читаемым: цвет muted, opacity не снижается.

Рабочая область до 1600px. Обзорный лабораторный экран до 1900px. На узком экране сайдбар уходит наверх.

Один главный действие на экран — `lab-btn-primary`. Остальные кнопки вторичные. В шапке карточки допустима тихая текстовая кнопка.

Пустое состояние — карточка: заголовок секции и одно предложение, что сделать дальше. Не иконка на полэкрана.

Вкладки — подчёркивание accent снизу, не пилюли. Контент вкладки начинается через 24px.

## Статус

Зелёный, жёлтый и красный — только здоровье, успех, предупреждение и ошибка. Информационный статус — accent.

Алерт, тост и карточка остаются нейтральными. Тон — иконка и левая граница 3px. Текст алерта всегда `--pdm-text`.

Прогресс: дорожка `--pdm-border-soft`, заполнение `--pdm-accent`, скругление на всю высоту. Высота в лаборатории не задана; в переносимом CSS — 8px.

Рабочий процесс, если он линейный: вертикальная нить 2px и точка 8px. Пройденное и текущее — accent. Текущий шаг дополнительно кольцо `0 0 0 3px` accent на 28%. Будущее — кольцо `--pdm-border-strong` и дырка цветом сайдбара.

## Графики

Один стилизатор на все фигуры. Готовые темы Plotly (`plotly`, `plotly_dark`, `ggplot2`) не использовать: `template = "none"` и явные токены.

| Параметр | Значение |
| --- | --- |
| Фон бумаги и области | `--pdm-chart-bg` |
| Шрифт | `--pdm-font`, 12px, `--pdm-chart-text` |
| Заголовок на графике | 13px, `--pdm-chart-text`. Лучше вынести заголовок в секцию над графиком |
| Поля | left 56, right 16, bottom 48, top 16; +32 если есть заголовок; top 48 если легенда видима |
| Легенда | горизонтально, сверху слева, без рамки и без фона |
| Hover | `x unified`; фон `--pdm-surface`, граница `--pdm-border`, текст `--pdm-text`, 12px |
| Ось X | линия 1px `--pdm-chart-axis`, сетки нет |
| Ось Y | линия 1px, сетка `--pdm-chart-grid` |
| Рамка виджета | поверхность, граница 1px, радиус 12 |

Роли рядов. Их не подменять циклической палитрой:

| Роль | Токен | Начертание |
| --- | --- | --- |
| Наблюденный сигнал | `--pdm-series-observed` | сплошная, 2px |
| Факт в будущем, если он показан как сверка | `--pdm-series-future-actual` | сплошная, тоньше или той же ширины, уже полупрозрачный |
| Прогноз | `--pdm-series-forecast` | сплошная, акцент |
| Коридор неопределённости | заливка `--pdm-series-band`, край `--pdm-series-band-line` | без отдельной легенды на каждую границу |
| Опорная линия «сейчас» / истина | `--pdm-series-reference` | точка 1px |
| Жёлтый и красный порог | `--pdm-zone-yellow`, `--pdm-zone-red` | штрих 1px |
| Заливка зон порога | `--pdm-zone-yellow-fill`, `--pdm-zone-red-fill` | под линиями, без обводки |
| Неизвестная зона | `--pdm-zone-unknown` | не путать с muted интерфейса |

Цикл `--pdm-series-1…6` — только когда на одном графике несколько равноправных каналов и у них нет ролей выше. Прогноз всегда акцент, даже если каналов много. Наблюдение всегда серое.

Зоны здоровья: green / yellow / red / unknown. Серый в мониторинге — это unknown, не «выключено».

Направление порога задаёт порядок заливок. Сигнал растёт вверх: жёлтая полоса между yellow и red, красная выше red. Сигнал падает вниз: красная ниже red, жёлтая между red и yellow.

Тепловая карта — только шкала из таблицы выше, от поверхности к акценту. Не rainbow.

## Текст интерфейса

Коротко и буквально. Одно предложение объясняет, что контрол меняет: данные, обучение или только отображение. Не писать, что лимит «улучшает модель», если он только красит результат.

Имена экранов и кнопок — обычный регистр предложений, не капс. Капс только у групповых меток сайдбара.

Числа с единицами измерения сигнала (`g`, `Pa`, `°C`). Единица не конвертирует значения, она подписывает ось.

Состояния задачи словами: `queued`, `running`, `stopping`, `completed`, `failed`, `stopped`, `idle`. Ошибку показывать текстом в алерте danger, не только кодом.

## Что держать одинаковым в обоих приложениях

- Тёмная тема по умолчанию и тот же переключатель Light / Dark.
- Те же hex и те же роли рядов на графиках.
- Один акцент. Второй акцент не вводить.
- Зелёный / жёлтый / красный не использовать для навигации, ссылок и обычных кнопок.
- Карточки плоские, радиус 12, граница 1px. Статус — полоска слева, не заливка.
- Заголовок 28 и одно предложение под ним.
- Сайдбар с группами и текущим пунктом на `--pdm-accent-soft`.

## Чего не делать

- Не брать цвета из `docs/wow_page_design.md`.
- Не красить прогноз оранжевым, а наблюдение голубым.
- Не включать сетку по X и не прятать ось.
- Не ставить легенду справа вертикально.
- Не рисовать градиентные кнопки, стеклянные панели и неоновое свечение.
- Не понижать opacity недоступного пункта навигации: он и так muted.
- Не подмешивать чужой шрифтовой стек. Inter допустим как запасной, не как отдельный характер.
