from itertools import groupby
import numba
import numpy as np
from numba import float64, int32
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
from jaxtyping import Int, Float, Scalar
ZERO = 1.0e-8




@numba.njit(cache=True)
def _anomaly_scores(
    X: np.ndarray,
    global_topic_prob: np.ndarray,
    mode_probs: list[np.ndarray],
    K: int,
    n_modes: int,
):
    ZERO = 1.0e-8
    time_size = global_topic_prob.shape[0]
    # n_modes = X.shape[1]
    scores = np.zeros(time_size)
    # event_nums = np.zeros(time_size)
    for x in X:
        val_ = ZERO
        for k in range(K):
            prob = global_topic_prob[x[0], k]
            kval_ = np.log(prob + ZERO)
            for m in range(1, n_modes):
                kval_ += np.log(mode_probs[m - 1][x[m], k] + ZERO)
            val_ = kval_ if k == 0 else np.logaddexp(val_, kval_)
        scores[x[0]] -= val_
        # event_nums[x[0]] += 1
    return scores


def plot_wordcloud(
    out_dir: Path,
    K: int,
    counterM: list[np.ndarray],
    mode_names: list[str],
    categories: list[str],
    n_mode_cate: int,
):
    for mode in range(1, n_mode_cate + 1):
        nrows = K // 4 + (K % 4 != 0)
        fig, axs = plt.subplots(ncols=4, nrows=nrows, figsize=(20, 15))
        axs = axs.flatten()

        def color_func(word, font_size, position, orientation, random_state, font_path):
            return "darkturquoise"

        for k in range(K):
            dic = _frequency_dic(categories[mode - 1], counterM[mode], mode, k)
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
                out_dir / f"mode{mode}_topic_{k + 1}.txt",
                mode="w",
            ) as f:
                for key in sorted_keys:
                    f.writelines(f"{key} : {dic[key]}\n")
        fig.tight_layout()
        fig.savefig(out_dir / f"{mode_names[mode - 1]}.png")


@numba.njit(int32(float64[:]))
def draw_one(posts: np.ndarray):
    residual = np.random.uniform(0, np.sum(posts))
    return_sample = 0
    for sample, prob in enumerate(posts):
        residual -= prob
        if residual < 0.0:
            return_sample = sample
            break
    return return_sample



def _frequency_dic(label_list: list[str], counterM: np.ndarray, mode: int, k: int):
    dic = {}
    counterM = counterM.astype(int)
    label_list = [str(label).split(".")[-1] for label in label_list]
    for i in range(counterM.shape[0]):
        dic[f"{label_list[i]}"] = counterM[i, k]
        assert counterM[i, k] != np.nan
    return dic
