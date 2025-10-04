import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from wordcloud import WordCloud

from _src.configs import DataConfig, ModelConfig

from .utils import _frequency_dic


class Basemodel:
    def __init__(
        self,
        tensor: pd.DataFrame,
        config: ModelConfig,
        dataConfig: DataConfig,
        categorical_idxs,
        keep_best_factors: bool,
        early_stoppping: bool,
    ):
        self.k = config.k
        """ number of topics/components"""
        self.n_dims: np.ndarray = tensor.max().values + 1
        """dimension of each attribute"""
        self.config: ModelConfig = config
        """configure class"""
        self.init_len = dataConfig.init_len
        self.anomaly = config.anomaly
        self.dataConfig: DataConfig = dataConfig
        self.time_idx = dataConfig.time_idx
        self.width = config.width
        self.n_dims[0] = config.width
        self.n_dims = self.n_dims.astype(int)
        self.n_modes: int = len(self.n_dims)
        """モード数"""
        self.logger = logging.getLogger(f"{__class__}")
        self.categorical_idxs: list[str] = categorical_idxs
        self.verbose: bool = config.verbose
        self.keep_best_factors: bool = keep_best_factors
        """whether to use the parameters from the best iteration."""
        self.early_stoppping: bool = early_stoppping
        """If performance drops during sampling, stop it."""
        self.regimes = []
        self.assignment_hist = []
        self.best_likelihoods = []

    def init_infer(self, *args) -> list[list[int]]:
        return

    def infer_online(self, tensor: pd.DataFrame, n_iter: int, *args):
        return

    def plot_online(self, out_dir: pathlib.Path, tensor: pd.DataFrame, timestamp: pd.DatetimeIndex):
        return

    def plot(
        self,
        out_dir: pathlib.Path,
        encoder: OrdinalEncoder,
        time_labels: pd.DatetimeIndex,
    ):
        categories = encoder.categories_

        for mode in range(1, self.n_modes):
            mode_dir = out_dir / f"{self.categorical_idxs[mode - 1]}"
            mode_dir.mkdir(exist_ok=True)
            nrows = self.k // 4 if self.k % 4 == 0 else self.k // 4 + 1
            for i, regime in enumerate(self.regimes):
                fig, axs = plt.subplots(ncols=4, nrows=nrows, figsize=(20, 15))
                axs = axs.flatten()

                def color_func(word, font_size, position, orientation, random_state, font_path):
                    return "darkturquoise"

                for k in range(self.k):
                    dic = _frequency_dic(categories[mode - 1], regime.counterM[mode], mode, k)
                    if sum(dic.values()) <= 0:
                        continue
                    im = WordCloud(
                        background_color="white",
                        color_func=color_func,
                        random_state=0,
                    ).generate_from_frequencies(dic)
                    axs[k].imshow(im)
                    axs[k].axis("off")
                    axs[k].set_title("Topic " + str(k + 1))
                    sorted_keys = sorted(dic, key=dic.get, reverse=True)
                    with open(
                        mode_dir / f"regime_{i + 1}_topic_{k + 1}.txt",
                        mode="w",
                    ) as f:
                        for key in sorted_keys:
                            f.writelines(f"{key} : {dic[key]}\n")
                plt.tight_layout()
                plt.savefig(mode_dir / f"regime_{i + 1}.png")

    def save_online(
        self,
        outdir: pathlib.Path,
        tensor: pd.DataFrame,
        encoder: OrdinalEncoder,
    ):
        return

    def save(self, outdir: pathlib.Path, tensor: pd.DataFrame, regime_assignments: list[tuple[int, int]], elapsed_times: list[float], *args):
        np.savetxt(outdir / "regime.txt", regime_assignments)
        np.savetxt(outdir / "elapsed_time.txt", np.array(elapsed_times))
        with open(outdir / "keys.csv", "w") as f:
            f.write(",".join(tensor.columns))

    def rgm_update_fin(self):
        return
