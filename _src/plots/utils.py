from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn import metrics
from wordcloud import WordCloud


def chunked_sum(x, chunk_size, axis=0):
    return np.array([x[i : i + chunk_size].sum(axis=axis) for i in range(0, len(x), chunk_size)])


def set_minor_tick_per_month(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%m",  # Changed to year-month format
    tick_collor: str = "gray",
    tick_linestyle: str = "--",
    show_labels: bool = True,
):
    """
    Sets the minor ticks on the x-axis to the beginning of each month.
    """
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(day=1, hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()

    # Use MonthLocator instead of DayLocator
    monthLocator = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(monthLocator)
    if show_labels:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(format))
    # Draw vertical lines at the beginning of each month
    for tick in monthLocator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.3)
    return start, end


def set_minor_tick(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    freq="H",
    format: str = "%H",
    tick_collor: str = "gray",
    tick_linestyle: str = "--",
    show_labels: bool = True,
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    minor_ticks = pd.date_range(start=start, end=end, freq=freq)
    minor_locator = ticker.FixedLocator(minor_ticks.map(mdates.date2num))
    ax.xaxis.set_minor_locator(minor_locator)
    for tick in minor_locator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    if show_labels:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(format))


def to_datetime(value: pd.Timestamp | np.datetime64) -> datetime:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    raise Exception


def set_major_tick_per_year(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%Y",
    tick_collor: str = "black",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    dayLocator = mdates.YearLocator()
    ax.xaxis.set_major_locator(dayLocator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    for tick in dayLocator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    _set_major_tick_pos(ax, label_loc, rotation)
    return start, end


def set_major_tick_per_day(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%m-%d",
    tick_collor: str = "black",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    dayLocator = mdates.DayLocator()
    ax.xaxis.set_major_locator(dayLocator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    for tick in dayLocator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    _set_major_tick_pos(ax, label_loc, rotation)
    return start, end


def _set_major_tick_pos(
    ax,
    label_loc: float = -0.03,
    rotation: int = 0,
):
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)  # rotate
        label.set_verticalalignment("top")
        label.set_y(label_loc)


def set_minor_tick_per_day(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    format: str = "%m-%d",
    tick_collor: str = "gray",
    tick_linestyle: str = "--",
    show_labels: bool = True,
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    dayLocator = mdates.DayLocator()
    ax.xaxis.set_minor_locator(dayLocator)
    if show_labels:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(format))
    for tick in dayLocator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.3)
    return start, end


def split_intervals(timestamps: np.ndarray, freq: str, skip_num: int = 0):
    match freq:
        case "S":
            threshold = np.timedelta64(10, "m")
        case "D":
            threshold = np.timedelta64(15, "D")
        case "H":
            threshold = np.timedelta64(48, "h")
    mask = np.diff(timestamps) >= threshold
    indices = np.where(mask)[0] + 1
    start = 0
    intervals = []
    widths = []
    for i, indice in enumerate(indices):
        if i < skip_num:
            start = indice
            continue
        intervals.append((start, indice))
        width = timestamps[indice - 1] - timestamps[start]
        if isinstance(width, np.timedelta64):
            width = width.item()
        elif isinstance(width, pd.Timedelta):
            width = width.to_numpy().item()
        widths.append(width)
        start = indice
    intervals.append((start, len(timestamps)))
    width = timestamps[-1] - timestamps[start]
    if isinstance(width, np.timedelta64):
        width = width.item()
    elif isinstance(width, pd.Timedelta):
        width = width.to_numpy().item()
    widths.append(width)
    return np.array(intervals), np.array(widths)


def _plot_wordcloud(
    ax: Axes,
    counterM: np.ndarray,
    categories: list[str],
    rgb_color: tuple[int] | None = None,
    max_word: int | None = None,
):
    def color_func(word, font_size, position, orientation, random_state, font_path):
        if rgb_color is not None:
            return rgb_color
        return "darkturquoise"

    dic = {}
    if max_word is None:
        for i in range(counterM.shape[0]):
            dic[f"{categories[i]}"] = counterM[i]
    else:
        for i in np.argsort(counterM)[::-1][:max_word]:
            dic[f"{categories[i]}"] = counterM[i]
    if sum(dic.values()) <= 0:
        return

    wc = WordCloud(
        background_color="white",
        color_func=color_func,
        random_state=0,
        min_font_size=12,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(dic)
    ax.imshow(wc, interpolation="bilinear", aspect="auto", extent=(0, wc.width, 0, wc.height))
