import logging
import pickle
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, auc, roc_auc_score

from _src import (
    Config,
    load_dataset,
    log_init,
    prepare_event_tensor,
)
from _src.plots import plot_anomalies, set_major_tick_per_day, set_minor_tick


def chunked_sum(x, chunk_size, axis=0):
    return np.array([x[i : i + chunk_size].sum(axis=axis) for i in range(0, len(x), chunk_size)])


def load_anomaly_idx(tensor: pd.DataFrame, config: Config, is_batch_wise: bool):
    anomaly_indexes = tensor.loc[tensor[config.data.label_col] != 0, config.data.time_idx].to_numpy()
    if is_batch_wise:
        anomaly_indexes //= config.model.width
    return np.unique(anomaly_indexes)


def plot_anomaly_scores(ax: Axes, scores: np.ndarray, time_index: pd.DatetimeIndex, true_label: np.ndarray):
    colors = sns.color_palette(n_colors=3)
    true_label = true_label.astype(int)
    labels = ["false", "true"]
    for i in [0, 1]:
        ax.scatter(
            time_index[true_label == i],
            scores[true_label == i],
            marker="*",
            c=colors[i],
            label=labels[i],
        )
    # ax.scatter(time_index, scores)
    ax.set_xlim(time_index.min(), time_index.max())
    ax.set_ylabel("anomaly score")
    set_major_tick_per_day(ax, time_index)
    set_minor_tick(ax, time_index, freq="8H")


def plot_pr_curve(fig_path: Path, true_labels: np.ndarray, anomaly_scores: np.ndarray):
    # Pr-auc直線
    precision, recall, thresholds = metrics.precision_recall_curve(true_labels, anomaly_scores)
    pr_auc = metrics.auc(recall, precision)
    plt.plot(recall, precision, label="PR curve (area = %.2f)" % pr_auc)
    plt.legend()
    plt.title("PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close("all")
    return pr_auc, precision, recall, thresholds


def load_animaly_score(out_dir: Path, config: Config, is_batchwise: bool):
    assert out_dir.exists()
    match config.model.name:
        case "heterocomp":
            anomalies = np.loadtxt(out_dir / "chisquare_scores.txt").astype(float)  # (batch_num, topic)
    return anomalies


def plot_anomaly_predict(
    path: Path,
    time_index: pd.DatetimeIndex,
    anomaly_idx: np.ndarray,
    true_label: np.ndarray,
    true_idx: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(30, 8))
    colors = sns.color_palette(n_colors=3)
    true_label = true_label.astype(int)
    labels = ["false", "true"]
    y = np.arange(anomaly_idx.shape[0])
    anomaly_idx = np.sort(anomaly_idx)
    for i in [0, 1]:
        idx = anomaly_idx[true_label[anomaly_idx] == i]
        ax.scatter(
            time_index[idx],
            y[true_label[anomaly_idx] == i],
            marker="*",
            c=colors[i],
            label=labels[i],
        )
    set_major_tick_per_day(ax, time_index)
    set_minor_tick(ax, time_index, freq="8H")
    ax.set_xlim(time_index.min(), time_index.max())
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_anomaly_true(
    path: Path,
    time_index: pd.DatetimeIndex,
    anomaly_idx: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(30, 8))
    ax.scatter(time_index[anomaly_idx], np.ones(anomaly_idx.shape[0]), marker="*")
    # ax.plot(time_index, anomalies)
    ax.set_xlim(time_index.min(), time_index.max())
    set_major_tick_per_day(ax, time_index)
    set_minor_tick(ax, time_index, freq="8H")
    ax.set_ylabel("anomaly score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@hydra.main(version_base=None, config_path="_src/configs", config_name="base_config")
def main(config: Config):
    try:
        log_init()
        IS_BATCH_WISE = True
        origin_stamps = None
        logger = logging.getLogger("main")
        seeds = [0, 10, 20] if config.data.seed == -1 else [config.data.seed]
        if config.model.name == "MStream":
            seeds = [0]
        for seed in seeds:
            print("===================================")
            logger.info(f"{seed=}")
            input_dir = Path(f"./_out/{config.model.name}")
            input_dir /= config.data.name
            config.data.seed = seed
            input_dir /= f"seed{config.data.seed}"
            input_dir /= f"topic{config.model.k}_scale{config.data.time_scale}_width{config.model.width}_initlen{config.data.init_len}"
            logger.info(f"{input_dir=}")
            assert input_dir.exists()
            time_idx = config.data.time_idx
            continuous_idxs = list(config.data.continuous_idxs)
            categorical_idxs = list(config.data.categorical_idxs)
            outputdir = input_dir / "anomaly"
            outputdir.mkdir(exist_ok=True)
            pickle_dir = Path("./_out/anomaly") / f"{config.data.name}" / "save.pkl"
            with open(pickle_dir, "rb") as f:
                anomalies, normal_counterM, normal_counterT = pickle.load(f)
            event_counterT = normal_counterT.copy()
            anom_counterT = np.zeros_like(normal_counterT)
            for anomaly in anomalies:
                anom_counterT += anomaly.counterT
                event_counterT += anomaly.counterT
            tensor = None
            if origin_stamps is None:
                time_dir = input_dir / "timestamps.txt"
                if time_dir.exists():
                    origin_stamps = pd.to_datetime(np.loadtxt(time_dir, dtype=str)).to_numpy()
                else:
                    raw_df = load_dataset(config.data.name, time_idx, continuous_idxs, categorical_idxs)
                    tensor, category_encoder, timepoint_encoder, origin_stamps, _ = prepare_event_tensor(
                        raw_df,
                        categorical_idxs,
                        time_idx,
                        freq=config.data.freq,
                        outdir=outputdir,
                    )
                    np.savetxt(time_dir, origin_stamps, fmt="%s")
            anomaly_scores = load_animaly_score(input_dir, config, IS_BATCH_WISE)
            if config.model.name in ("MemStream", "ARCUS", "MStream"):
                score_path = input_dir / "score_time.txt"
                if score_path.exists():
                    anomaly_scores = np.loadtxt(score_path)
                else:
                    if tensor is None:
                        raw_df = load_dataset(config.data.name, time_idx, continuous_idxs, categorical_idxs)
                        tensor, category_encoder, timepoint_encoder, origin_stamps, _ = prepare_event_tensor(
                            raw_df,
                            categorical_idxs,
                            time_idx,
                            freq=config.data.freq,
                            outdir=outputdir,
                        )
                    tensor["score"] = anomaly_scores
                    anomaly_scores = tensor.groupby(time_idx)["score"].sum()
                    np.savetxt(score_path, anomaly_scores)
                anomaly_scores = chunked_sum(anomaly_scores, config.model.width)

            # true label
            if IS_BATCH_WISE:
                timestamps = origin_stamps[:: config.model.width]
                anom_counterT = chunked_sum(anom_counterT, config.model.width)
                event_counterT = chunked_sum(event_counterT, config.model.width)
            else:
                timestamps = origin_stamps
            anomaly_indexes = np.where(anom_counterT >= 100)[0]

            plot_anomaly_true(outputdir / "anomaly_true.png", timestamps, anomaly_indexes)
            logger.info(f"{timestamps.shape=}")
            true_labels = np.zeros_like(anomaly_scores)
            for idx in anomaly_indexes:
                true_labels[idx] = 1

            # plot anomaly score
            fig, ax = plt.subplots(2, 1, figsize=(30, 8))
            ax = ax.flatten()
            plot_anomaly_scores(ax[0], anomaly_scores, timestamps, true_labels)
            start = pd.to_datetime(timestamps[0]).replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
            end = pd.to_datetime(timestamps[-1]).to_pydatetime()
            colors = sns.color_palette("hls", len(anomalies) + 1)
            plot_anomalies(ax[1], colors, anomalies, start, end, 8)
            ax[1].set_xlim([timestamps.min(), timestamps.max()])
            fig.tight_layout()
            fig.savefig(outputdir / "anomaly_scores.pdf")
            plt.close(fig)

            roc_auc = roc_auc_score(true_labels, anomaly_scores)
            logger.info(f"ROC-AUC score: {roc_auc:.4f}")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(true_labels, anomaly_scores, ax=ax)
            fig.tight_layout()
            fig.savefig(outputdir / "auc-roc.png")
            plt.close(fig)

            # pr曲線
            pr_auc, precision, recall, thresholds = plot_pr_curve(outputdir / "pr-auc.png", true_labels, anomaly_scores)
            logger.info(f"PR-AUC score: {pr_auc:.4f}")

            with open(outputdir / "anomaly_result.txt", "w") as f:
                f.write(f"pr-auc: {pr_auc}\n")
                f.write(f"roc-auc: {roc_auc}\n")
            # plot top k
            anomaly_sort_idx = np.argsort(anomaly_scores)
            for top_k in [10, 100, 300, 500, 1000]:
                top_k_idx = anomaly_sort_idx[-top_k:]
                plot_anomaly_predict(
                    outputdir / f"anomaly_top{top_k}.png",
                    timestamps,
                    top_k_idx,
                    true_labels,
                    anomaly_indexes,
                )

    except Exception as ex:
        logger.exception(ex)


main()
