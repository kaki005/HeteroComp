import logging
import pickle
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

from _src import (
    Config,
    load_dataset,
    log_init,
    prepare_event_tensor,
    tree_patrition,
)
from _src.plots import chunked_sum, extract_anomalies, extract_anomalies_ci, plot_anomalies, plot_anomaly_time_hist, plot_counter, plot_time_histgram


@hydra.main(version_base=None, config_path="_src/configs", config_name="base_config")
def main(config: Config):
    try:
        log_init()
        logger = logging.getLogger("main")
        out_dir = Path("./_out/anomaly") / f"{config.data.name}"
        out_dir.mkdir(exist_ok=True, parents=True)

        cont_idx = list(config.data.continuous_idxs)
        cate_idx = list(config.data.categorical_idxs)
        raw_df = load_dataset(config.data.name, config.data.time_idx, cont_idx, cate_idx)
        raw_df, oe, timepoint_encoder, timestamps, sorted_indice = prepare_event_tensor(raw_df, cate_idx, config.data.time_idx, config.data.freq, out_dir)
        mode_bounds = []
        for idx in cont_idx:
            bounds, counter = tree_patrition(raw_df[idx].to_numpy(), config.model.num_bins, min_points=10)
            bins = np.hstack([bounds[:, 0] - 1e-10, bounds[-1, 1] + 1e-10])
            binned_df = pd.cut(
                raw_df[idx],
                bins=bins,
                labels=False,
            )
            raw_df[idx] = binned_df.astype(int)
            mode_bounds.append(bounds)
        logger.info(f"{pd.unique(raw_df[config.data.label_col])=}")
        start = pd.to_datetime(timestamps[0]).replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
        end = pd.to_datetime(timestamps[-1]).to_pydatetime()
        if config.data.name in ("cci18", "ci17"):
            raw_df.loc[raw_df["Label"] == "Benign", "Label"] = 0
            raw_df.loc[raw_df["Label"] == "BENIGN", "Label"] = 0
        time_dir = out_dir / "timestamps.txt"
        np.savetxt(time_dir, timestamps, fmt="%s")
        logger.info(f"mode bounds.shape = {[bound.shape for bound in mode_bounds]}")
        n_dims = raw_df[cate_idx + cont_idx].max().to_numpy() + 1
        logger.info(f"{n_dims=}")

        # =====================
        # plot
        # =====================
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        pickle_dir = out_dir / "save.pkl"
        if pickle_dir.exists():
            with open(pickle_dir, "rb") as f:
                anomalies, normal_counterM, normal_counterT = pickle.load(f)
        else:
            if config.data.name in ("ci17", "cci18", "DDoS2019", "CUPID", "Edge", "WebIDS2023"):
                anomalies, normal_counterM, normal_counterT = extract_anomalies_ci(raw_df, config.data, timestamps)
            else:
                anomalies, normal_counterM, normal_counterT = extract_anomalies(raw_df, config.data, timestamps)
            with open(pickle_dir, "wb") as f:
                pickle.dump((anomalies, normal_counterM, normal_counterT), f)

        colors = sns.color_palette("hls", len(anomalies) + 1)
        plot_anomalies(ax, colors, anomalies, start, end, 4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{config.data.name}_attack.png")
        plt.close(fig)

        # time histgram
        plot_anomaly_time_hist(out_dir, colors, anomalies, timestamps, config.data.freq, 4)
        plot_time_histgram(out_dir / "normal_histgram.png", [colors[-1]], normal_counterT, timestamps, ["normal"], config.data.freq)
        for width in [30, 60, 120, 240]:
            chunk_normal_counterT = chunked_sum(normal_counterT, width)
            plot_time_histgram(out_dir / f"normal_hist_{width}_{config.data.freq}.png", [colors[-1]], chunk_normal_counterT, timestamps[::width], ["normal"], config.data.freq)
            full_counterT = np.array([chunked_sum(anomaly.counterT, width) for anomaly in anomalies] + [chunk_normal_counterT]).T
            labels = [anomaly.attack_name for anomaly in anomalies] + ["normal"]
            plot_time_histgram(out_dir / f"full_hist_{width}_{config.data.freq}.png", colors, full_counterT, timestamps[::width], labels, config.data.freq)

        plot_counter(out_dir, config.data, mode_bounds, normal_counterM, oe)
        for anomaly in anomalies:
            anomaly.plot_counter(out_dir, mode_bounds, oe)

    except Exception as ex:
        logger.exception(ex)
        sys.exit(-1)


main()
