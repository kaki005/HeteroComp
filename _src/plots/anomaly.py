import datetime
import logging
import os
from datetime import datetime
from pathlib import Path
from turtle import width

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from _src.configs import DataConfig

from .utils import _plot_wordcloud, set_major_tick_per_day, set_major_tick_per_year, set_minor_tick, set_minor_tick_per_month, split_intervals, to_datetime


class Anomaly:
    def __init__(
        self,
        n_dims: np.ndarray,
        start: datetime,
        end: datetime,
        attack_name: str,
        id: int,
        cont_idx: list[str],
        cate_idx: list[str],
        counterM: dict[str, np.ndarray],
        counterT: np.ndarray,
        data_indexes: list[int],
    ):
        self.start = start
        self.end = end
        self.cont_idx = cont_idx
        self.cate_idx = cate_idx
        self.cont_num = len(cont_idx)
        self.cate_num = len(cate_idx)
        self.attack_name = attack_name
        self.n_dims = n_dims
        self.id = id
        self.counterM = counterM
        """(categorical + continuous)"""
        self.counterT = counterT
        self.logger = logging.getLogger()
        self.data_indexes: list[int] = data_indexes
        """data indexes"""

    @property
    def mode_num(self) -> int:
        return self.cont_num + self.cate_num

    @property
    def num(self) -> int:
        return np.sum(self.counterM[self.cate_idx[0]])

    def plot_counter(self, out_dir: Path, mode_bounds: list[np.ndarray], encoder: OrdinalEncoder):
        # counter
        fig, axes = plt.subplots(self.cont_num, 1, figsize=(90, 7 * self.cont_num))
        if self.cont_num == 1:
            axes = [axes]
        for i, idx in enumerate(self.cont_idx):
            bin_x = np.mean(mode_bounds[i], axis=1).squeeze()
            axes[i].bar(np.arange(bin_x.shape[0]), self.counterM[idx])
            axes[i].set_xlabel(idx)
            axes[i].set_xlim(-1, bin_x.shape[0] + 1)
            axes[i].set_xticks(np.arange(bin_x.shape[0]))
            axes[i].set_xticklabels([f"{x:.2e}" for x in bin_x])
            for label in axes[i].get_xticklabels():  # rotate label
                label.set_rotation(90)
        fig.tight_layout()
        fig.savefig(out_dir / f"{self.id}_{self.attack_name}.png")
        plt.close(fig)
        with open(out_dir / f"{self.id}_{self.attack_name}_count.txt", "w") as f:
            for i, idx in enumerate(self.cate_idx):
                f.write(f"\n{idx}\n")
                for index in np.argsort(self.counterM[idx])[::-1]:
                    if self.counterM[idx][index] <= 0:
                        break
                    f.write(f"{encoder.categories_[i][index]} : {self.counterM[idx][index]}\n")
        # word cloud
        ncols = 3
        nrows = self.cate_num // ncols + (self.cate_num % ncols != 0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))
        axes = axes.flatten()
        for i, idx in enumerate(self.cate_idx):
            _plot_wordcloud(axes[i], self.counterM[idx], encoder.categories_[i])
            axes[i].set_xlabel(idx)
        fig.tight_layout()
        fig.savefig(out_dir / f"{self.id}_{self.attack_name}_wordcloud.png")
        plt.close(fig)


def plot_counter(
    out_dir: Path,
    config: DataConfig,
    mode_bounds: list[np.ndarray],
    counterM: dict[str, np.ndarray],
    encoder: OrdinalEncoder,
):
    cont_idx = list(config.continuous_idxs)
    cate_idx = list(config.categorical_idxs)
    cate_num = len(cate_idx)
    cont_num = len(cont_idx)
    fig, axes = plt.subplots(cont_num, 1, figsize=(90, 7 * cont_num))
    if cont_num == 1:
        axes = [axes]
    for i, idx in enumerate(cont_idx):
        logging.info(f"{idx}: {mode_bounds[i].shape=} {counterM[idx].shape=} ")
        bin_x = np.mean(mode_bounds[i], axis=1).squeeze()
        axes[i].bar(np.arange(bin_x.shape[0]), counterM[idx])
        axes[i].set_xlabel(idx)
        axes[i].set_xlim(-1, bin_x.shape[0] + 1)
        axes[i].set_xticks(np.arange(bin_x.shape[0]))
        axes[i].set_xticklabels([f"{x:.2e}" for x in bin_x])
        for label in axes[i].get_xticklabels():  # rotate label
            label.set_rotation(90)
    fig.tight_layout()
    fig.savefig(out_dir / "normal_counter.png")
    plt.close(fig)
    # counter list
    with open(out_dir / "normal_count.txt", "w") as f:
        for i, idx in enumerate(cate_idx):
            f.write(f"\n{idx}\n")
            for index in np.argsort(counterM[idx])[::-1]:
                if counterM[idx][index] <= 0:
                    break
                f.write(f"{encoder.categories_[i][index]} : {counterM[idx][index]}\n")

    # word cloud
    ncols = 3
    nrows = cate_num // ncols + (cate_num % ncols != 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))
    axes = axes.flatten()
    for i, idx in enumerate(cate_idx):
        _plot_wordcloud(axes[i], counterM[idx], encoder.categories_[i])
        axes[i].set_xlabel(idx)
    fig.tight_layout()
    fig.savefig(out_dir / "normal_wordclud.png")
    plt.close(fig)


def extract_anomalies_ci(
    df: pd.DataFrame,
    config: DataConfig,
    timestamps,
) -> tuple[list[Anomaly], dict[str, np.ndarray], np.ndarray]:
    attack_id = 0
    cont_idx = list(config.continuous_idxs)
    cate_idx = list(config.categorical_idxs)
    df[cont_idx + cate_idx] = df[cont_idx + cate_idx].astype(float).astype(int)
    n_dims = df[cate_idx + cont_idx].max().to_numpy() + 1
    label_col = config.label_col
    time_idx = config.time_idx
    T = timestamps.shape[0]
    anomaly_dic: dict[str, Anomaly] = {}
    normal_counterM = {idx: np.zeros(df[idx].max() + 1) for idx in cont_idx + cate_idx}
    normal_counterT = np.zeros(T, dtype=int)
    for i, row in df.iterrows():
        attack = row[label_col]
        if attack != 0:  # if anomaly
            if attack in anomaly_dic:
                anomaly_dic[attack].end = to_datetime(timestamps[row[time_idx]])  # update abnormal termination time
            else:
                counterM = {idx: np.zeros(df[idx].max() + 1) for idx in cont_idx + cate_idx}
                anomaly_dic[attack] = Anomaly(
                    n_dims, to_datetime(timestamps[row[time_idx]]), to_datetime(timestamps[row[time_idx]]), attack, attack_id, cont_idx, cate_idx, counterM, np.zeros(T, dtype=int), []
                )
                attack_id += 1  # 新しい異常id
            for j, idx in enumerate(cate_idx + cont_idx):
                anomaly_dic[attack].counterM[idx][row[idx]] += 1
            anomaly_dic[attack].counterT[row[time_idx]] += 1
            anomaly_dic[attack].data_indexes.append(i)
        else:
            for j, idx in enumerate(cate_idx + cont_idx):
                normal_counterM[idx][row[idx]] += 1
            normal_counterT[row[time_idx]] += 1
    return list(anomaly_dic.values()), normal_counterM, normal_counterT


def extract_anomalies(df: pd.DataFrame, config: DataConfig, timestamps: np.ndarray) -> tuple[list[Anomaly], dict[str, np.ndarray], np.ndarray]:
    anomalies = []
    start = None
    end = None
    attack_id = 0
    attack_name = ""
    T = timestamps.shape[0]
    cont_idx = list(config.continuous_idxs)
    cate_idx = list(config.categorical_idxs)
    df[cont_idx + cate_idx] = df[cont_idx + cate_idx].astype(float).astype(int)
    n_dims = df[cate_idx + cont_idx].max().to_numpy() + 1
    label_col = config.label_col
    time_idx = config.time_idx
    counterM = {idx: np.zeros(df[idx].max() + 1) for idx in cont_idx + cate_idx}
    counterT = np.zeros(T, dtype=int)
    normal_counterM = {idx: np.zeros(df[idx].max() + 1) for idx in cont_idx + cate_idx}
    normal_counterT = np.zeros(T, dtype=int)
    data_indexes = []
    for i, row in df.iterrows():
        attack = row[label_col]
        if attack != 0:  # 異常なら
            changeAnomaly = attack_name != attack
            data_indexes.append(i)
            if start is None or changeAnomaly:  # startがあり別の異常なら
                if start is not None:  # startがあり別の異常なら
                    if end is None:  # endがなければ今をendとする。
                        end = start
                    logging.info(f"{attack_id} : {start}~{end} {attack_name}")
                    anomalies.append(Anomaly(n_dims, start, end, attack_name, attack_id, cont_idx, cate_idx, counterM, counterT, data_indexes))
                    attack_id += 1  # 新しい異常id
                    counterM = {idx: np.zeros(df[idx].max() + 1) for idx in cont_idx + cate_idx}
                    counterT = np.zeros(T, dtype=int)
                    data_indexes = []
                start = to_datetime(timestamps[row[time_idx]])  # new anomaly
                attack_name = attack
                end = None
            else:
                end = to_datetime(timestamps[row[time_idx]])  # lat occurenct time
            for i, idx in enumerate(cate_idx + cont_idx):
                counterM[idx][row[idx]] += 1
            counterT[row[time_idx]] += 1
        else:  # 正常なら
            for i, idx in enumerate(cate_idx + cont_idx):
                normal_counterM[idx][row[idx]] += 1
            normal_counterT[row[time_idx]] += 1

    if start is not None:
        if end is None:
            end = start
        print(f"{attack_id} : {start}~{end} {attack_name}")
        anomalies.append(Anomaly(n_dims, start, end, attack_name, attack_id, cont_idx, cate_idx, counterM, counterT, data_indexes))
    return anomalies, normal_counterM, normal_counterT


def plot_anomalies(
    ax: Axes,
    colors,
    anomalies: list[Anomaly],
    start: pd.Timestamp,
    end: pd.Timestamp,
    label_interval: int = 1,
):
    for anomaly in anomalies:
        ax.plot(
            [anomaly.start, anomaly.end],
            [anomaly.id, anomaly.id],
            marker="*",
            color=colors[anomaly.id],
            markersize=10,
            label=anomaly.attack_name,
        )  #
        ax.text(
            anomaly.start,
            anomaly.id + 0.1,
            f"({anomaly.id}) {anomaly.attack_name} ({anomaly.start.strftime('%d day %H:%M:%S')}~{anomaly.end.strftime('%d day%H:%M:%S')}) : {anomaly.num}",
            fontsize=12,
        )
    set_major_tick_per_day(ax, start=start, end=end)
    set_minor_tick(ax, start=start, end=end, freq=f"{label_interval}H")
    # ax.legend()
    ax.set_ylabel("activityID")


def plot_time_histgram(img_path: Path, colors, counter: np.ndarray, timestamps: np.ndarray, labels, freq: str, label_interval: int = 1, ax_interval: float = 0.1):
    intervals, widths = split_intervals(timestamps, freq)
    intervals = intervals[widths > 1]
    widths = widths[widths > 1]
    if len(intervals) == 0:
        logging.info(f"{timestamps=}")
        return
    fig, axes = plt.subplots(
        ncols=len(intervals),
        figsize=(60, 8),
        gridspec_kw={"width_ratios": widths, "wspace": ax_interval},  # Adjust the horizontal spacing between axes in wspace)
        constrained_layout=True,
    )
    if len(intervals) == 1:
        axes = [axes]
    for i, (start, end) in enumerate(intervals):
        timestamps_i = pd.to_datetime(timestamps[start:end]).to_pydatetime()
        max_width = np.min(np.diff(timestamps_i))
        bar_width = np.minimum(np.diff(timestamps_i, append=timestamps_i[-1] + max_width), max_width)
        if counter.ndim == 1:
            mask = counter[start:end] > 0
            axes[i].bar(timestamps_i[mask], counter[start:end][mask], color=colors[0], label=labels[0], width=bar_width[mask])
            axes[i].set_ylim(0, counter.max() + 1)
        elif counter.ndim == 2:
            offset = np.zeros(timestamps_i.shape[0])
            for j in range(counter.shape[1]):
                if np.sum(counter[start:end, j]) <= 0:
                    continue
                mask = counter[start:end, j] > 0
                axes[i].bar(pd.Series(timestamps_i[mask]), counter[start:end, j][mask], color=colors[j], label=labels[j], bottom=offset[mask], width=bar_width[mask])
                offset += counter[start:end, j]
            axes[i].set_ylim(0, np.sum(counter, axis=1).max() + 1)
        axes[i].set_xlim(timestamps_i[0], timestamps_i[-1])
        match freq:
            case "D" | "H":
                set_major_tick_per_year(axes[i], timestamps_i)
                set_minor_tick_per_month(axes[i], timestamps_i)
            case "S":
                set_major_tick_per_day(axes[i], timestamps_i)
                set_minor_tick(axes[i], timestamps_i, freq=f"{label_interval}H")
        if i == 0:
            axes[i].set_ylabel("count")
        else:
            axes[i].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    axes[-1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # Shift to the right (x > 1.0 for the outer region)
        borderaxespad=0,
    )
    plt.subplots_adjust(right=0.8)  # legend
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def plot_anomaly_time_hist(out_dir: Path, colors, anomalies: list[Anomaly], timestamps: np.ndarray, freq: str, label_interval: int = 1):
    intervals, widths = split_intervals(timestamps, freq)
    mask = widths > 1
    intervals = intervals[mask]
    widths = widths[mask]
    gridspec_kw = {}
    if len(intervals) > 1:
        gridspec_kw = {"width_ratios": widths, "wspace": 0.1}  #
    # maxY = 0
    fig, axes = plt.subplots(
        ncols=len(intervals),
        figsize=(60, 8),
        gridspec_kw=gridspec_kw,
        constrained_layout=True,  # tight_layout
    )
    if len(intervals) == 1:
        axes = [axes]
    for i, (start, end) in enumerate(intervals):
        timestamps_i = timestamps[start:end]
        offset = np.zeros(timestamps_i.shape[0], dtype=int)
        # bar_width = np.minimum(np.diff(timestamps_i, append=timestamps_i[-1] + np.timedelta64(1, "s")), np.timedelta64(10, "s"))
        bar_width = np.diff(timestamps_i, append=timestamps_i[-1] + np.timedelta64(1, "s"))
        maxY = 10
        for j, anomaly in enumerate(anomalies):
            if anomaly.start < to_datetime(timestamps_i[0]) or anomaly.end >= to_datetime(timestamps_i[-1]):
                continue
            mask = anomaly.counterT[start:end] > 0
            axes[i].bar(timestamps_i[mask], anomaly.counterT[start:end][mask], color=colors[j], bottom=offset[mask], label=anomaly.attack_name, width=bar_width[mask])
            offset += anomaly.counterT[start:end]
            maxY = np.maximum(maxY, np.max(anomaly.counterT[start:end]))
        set_major_tick_per_day(axes[i], timestamps_i)
        set_minor_tick(axes[i], timestamps_i, freq=f"{label_interval}H")
        axes[i].set_xlim(timestamps_i[0], timestamps_i[-1])
        axes[i].set_ylim(0, maxY)
        axes[i].legend()
    fig.savefig(out_dir / "anomaly_histgram.png")
    plt.close(fig)
