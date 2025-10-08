"""Python implementation of CubeScope"""

import copy
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns
from jaxtyping import Float, Int
from scipy.stats import chi2
from sklearn.preprocessing import OrdinalEncoder

from _src.configs import DataConfig, ModelConfig
from _src.mode import (
    A_Polya,
    B_gp,
    Base,
    C_gp,
    datetime_base,
    freq_to_timedelta64,
)
from _src.plots import plot_time_histgram, split_intervals, to_datetime

from .basemodel import Basemodel
from .utils import draw_one

ZERO = 1.0e-8


# ===========================
# region(HeteroComp)
# ===========================


class HeteroComp(Basemodel):
    def __init__(
        self,
        tensor: pd.DataFrame,
        config: ModelConfig,
        dataConfig: DataConfig,
        mode_boundss: list[np.ndarray],
        keep_best_factors: bool = True,
        early_stoppping: bool = False,
        tensor_shape=[],
    ):
        super(HeteroComp, self).__init__(
            tensor,
            config,
            dataConfig,
            dataConfig.categorical_idxs,
            keep_best_factors,
            early_stoppping,
        )
        self.continuous_idxs = list(dataConfig.continuous_idxs)
        self.n_dims = np.full(1 + len(dataConfig.categorical_idxs) + len(dataConfig.continuous_idxs), 2)
        self.n_dims[0] = config.width
        for i, cate in enumerate(self.categorical_idxs):
            self.n_dims[1 + i] = tensor[cate].max() + 1
        for i in range(len(dataConfig.continuous_idxs)):
            self.n_dims[1 + self.nmode_cate + i] = mode_boundss[i].shape[0]
        self.n_dims = self.n_dims.astype(int)
        self.counterM: list[np.ndarray] = []
        """counter (attribute, dim, K)"""
        self.counterK: np.ndarray = np.zeros(self.k, dtype=int)
        """counter of component (K)"""
        self.counterA: np.ndarray = np.zeros(self.n_dims[0], dtype=int)
        """counter of time (T_c)"""
        self.colors: sns.palettes._ColorPalette = sns.color_palette("hls", self.k + 1)
        self.B: B_gp = B_gp(self.width, self.k, config, dataConfig, self.colors)
        self.categoricalA: list[A_Polya] = [A_Polya(i, self.n_dims[i + 1], self.k, config.alpha, config, idx) for i, idx in enumerate(list(dataConfig.categorical_idxs))]
        self.last_timestamps: np.ndarray | None = None
        """last time of previous tensor"""
        self.hist_counterTK = []
        """(bach, T_c, K)"""
        self.chisquare_scores = []
        self.detected_times = []
        self.continuousC: list[C_gp] = [C_gp(idx, self.k, config, dataConfig, mode_boundss[i], self.colors) for i, idx in enumerate(list(dataConfig.continuous_idxs))]
        self.assignment_hist = []
        """component assignment"""
        self.counterM_hist = []
        self.full_counterM = [np.zeros((dim, self.k)) for dim in self.n_dims[1:]]
        self.normal_counterM = [np.zeros((dim, self.k)) for dim in self.n_dims[1:]]
        self.total_interval = np.zeros(1)
        self.normal_counterK = np.zeros(self.k)
        self.DoF = self.k - 1  # degree of freedom
        """degree of freedom"""
        for mode in range(self.nmode_cate + self.nmode_cont):
            self.DoF += (self.n_dims[mode + 1] - 1) * self.k
        self.logger.info(f"{self.DoF=}")

    @property
    def modes(self) -> list[Base]:
        return self.categoricalA + self.continuousC

    @property
    def nmode_cate(self) -> int:
        return len(self.categorical_idxs)

    @property
    def nmode_cont(self) -> int:
        return len(self.continuous_idxs)

    def init_infer(self, tensor_train: pd.DataFrame, train_timestamps: np.ndarray, n_iter: int = 50) -> list[list[int]]:
        """Initialize model parameters i.e, training process
        1. batch estimation for each subtensor
        2. Initialize model parameters employing subtensors given by 1.
        """
        self.l = int((tensor_train[self.time_idx].max() + 1) // self.width)
        for mode in self.modes:
            mode.setL(self.l)
        self.B.setL(1)
        self.logger.info(f"{self.l=}")
        self.counterA = np.zeros(pd.unique(tensor_train[self.time_idx]).shape[0], dtype=int)
        each_samp_llh = []
        best_llh = -np.inf
        self.B.init_gibbs_batch(0, tensor_train, train_timestamps)
        for mode in self.modes:
            tensor_train = mode.init_gibbs_batch(0, tensor_train)
        assignment, n_events, self.counterM = self._init_status(tensor_train)
        self.counterM[0] = np.zeros((train_timestamps.shape[0], self.k), dtype=int)
        tensor_numpy = tensor_train.to_numpy()

        for iter_ in range(n_iter):
            assignment, changed_num = _gibbs_sampling(
                tensor_numpy,
                assignment,
                self.counterM,
                self.counterK,
                self.k,
                [a.alpha for a in self.categoricalA],
                self.B.means,
                self.B.variances,
                [c.mode_prob for c in self.continuousC],
                len(self.dataConfig.categorical_idxs),
                len(self.dataConfig.continuous_idxs),
            )
            self.B.post_gibbs_batch(0, self.counterM[0], self.counterA)
            llh = self.log_likelihood_init()
            if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                self.logger.info(f"llh_{iter_ + 1}: {llh}")
                self.logger.info(f"counterK = {self.counterK.tolist()}")
            each_samp_llh.append(llh)
            if (self.early_stoppping) and len(each_samp_llh) > 1 and (each_samp_llh[-2] > each_samp_llh[-1]):
                break
            if (self.keep_best_factors) and (llh > best_llh):
                best_counterM = copy.deepcopy(self.counterM)
                best_assignment = copy.deepcopy(assignment)
                best_counterK = copy.deepcopy(self.counterK)
                best_llh = llh
            if changed_num / n_events <= 0.01:
                self.logger.info(f"llh_{iter_ + 1}: {llh}")
                self.logger.info(f"counterK = {self.counterK.tolist()}")
                break
            if (iter_ + 1) % 3 == 0:
                self.B.update_hyperparam()
        if self.keep_best_factors:
            self.counterM = best_counterM
            self.counterK = best_counterK
            assignment = best_assignment
            self.logger.info(f"best llh: {best_llh}")
            self.logger.info(f"counterK = {self.counterK.tolist()}")
        for i, C_mode in enumerate(self.continuousC):
            C_mode.update_prev_dist(self.counterM[i + 1 + self.nmode_cate], self.counterK)
            C_mode.update_hyperparam(self.counterM[i + 1 + self.nmode_cate], self.counterK)
        # anomaly detection
        if self.anomaly:
            self.normal_counterM = [self.counterM[mode + 1] for mode in range(self.n_modes - 1)]
            self.normal_counterK = self.counterK
            self.total_interval = self._time_interval(train_timestamps)
        self.B.last_date = None  # reset
        n = 0
        for i, A_mode in enumerate(self.categoricalA):
            A_mode.reset()  # reset alpha
        for l in range(self.l):
            tensor = tensor_train[(tensor_train[self.time_idx] >= l * self.width) & (tensor_train[self.time_idx] < (l + 1) * self.width)]
            tensor.loc[:, self.time_idx] -= l * self.width
            _, _, counterM = self._init_status(tensor)
            for topic, x in zip(assignment[n : n + len(tensor)], tensor.to_numpy()):
                self.counterK[topic] += 1
                for mode in range(self.nmode_cate + 1):
                    counterM[mode][x[mode], topic] += 1
            n += len(tensor)
            for mode, a in enumerate(self.categoricalA):
                a.init_prev_dist(l, counterM[mode + 1], self.counterK)
        return [[0, 0]]  # initialize regime assignment

    def infer_online(
        self,
        tensor: pd.DataFrame,
        n_iter: int,
        timestamps: np.ndarray,
    ):
        self.counterA = np.zeros(self.n_dims[0], dtype=int)
        self.logger.info(f"# of events: {len(tensor)}")
        each_samp_llh = []  # log likelihood
        best_llh = -np.inf
        pre_timestamp = self.B.last_date
        if pre_timestamp is None:
            pre_timestamp = timestamps[0] - freq_to_timedelta64(self.dataConfig.freq)
        # inference
        self.B.init_gibbs(tensor, timestamps)
        for mode in self.modes:
            tensor = mode.init_gibbs(tensor)
        tensor_numpy = tensor.to_numpy()
        assignment, n_events, self.counterM = self._init_status(tensor)
        for iter_ in range(n_iter):
            assignment, changed_num = _gibbs_sampling_online(
                tensor_numpy,
                assignment,
                self.counterM,
                self.counterK,
                self.k,
                [a.alpha for a in self.categoricalA],
                self.B.means,
                self.B.variances,
                self.l,
                [a.prev_terms for a in self.categoricalA],
                [c.mode_prob for c in self.continuousC],
                len(self.dataConfig.categorical_idxs),
                len(self.dataConfig.continuous_idxs),
            )
            self.B.post_gibbs(self.counterM[0], self.counterA)
            llh = self.log_likelihood()
            if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                self.logger.info(f"llh_{iter_ + 1}: {llh}")
                self.logger.info(f"counterK = {self.counterK.tolist()}")
            each_samp_llh.append(llh)
            if (self.early_stoppping) and len(each_samp_llh) > 1 and (each_samp_llh[-2] > each_samp_llh[-1]):
                break
            if (self.keep_best_factors) and (llh > best_llh):
                best_counterM = copy.deepcopy(self.counterM)
                best_counterK = copy.deepcopy(self.counterK)
                best_assignment = copy.deepcopy(assignment)
                best_llh = llh
            if changed_num / n_events < 0.05:  # if burn-in
                break
        if self.keep_best_factors:
            self.counterM = best_counterM
            self.counterK = best_counterK
            assignment = best_assignment
            self.best_likelihoods.append(best_llh)
            self.logger.info(f"best llh: {best_llh}")
        # anormaly detection
        if self.anomaly:
            self._calc_anomaly_score(tensor, timestamps, assignment, self.counterM, self.counterK, pre_timestamp, save_score=True)
        self.assignment_hist += list(assignment)
        self.hist_counterTK.append(self.counterM[0])
        # self.counterM_hist.append(self.counterM)
        for mode in range(self.n_modes - 1):
            self.full_counterM[mode] += self.counterM[mode + 1]
        # update parameter:
        # categorical
        for mode, a in enumerate(self.categoricalA):
            a.update_prev_dist(self.counterM[mode + 1], self.counterK)
        # continuous
        for mode, C_mode in enumerate(self.continuousC):
            C_mode.counterQueue.append(self.counterM[mode + 1 + self.nmode_cate])
            if len(C_mode.counterQueue) > self.l:
                C_mode.counterQueue.popleft()  # pop old counter
            counterM = np.sum(list(C_mode.counterQueue), axis=0)
            C_mode.update_prev_dist(counterM, np.sum(counterM, axis=0))
        self.B.update_prev_dist(self.counterM[0], self.counterA)
        for mode in self.modes:
            mode.save_history()
        self.B.save_history()
        return False

    def save_online(
        self,
        outdir: Path,
        tensor: pd.DataFrame,
        encoder: OrdinalEncoder,
    ):
        for mode in range(self.nmode_cate):
            self.categoricalA[mode].save_online(outdir, self.counterM[mode + 1], encoder.categories_[mode])
        for mode in range(self.nmode_cont):
            self.continuousC[mode].save_online(outdir, self.counterM[self.nmode_cate + mode + 1])

    def save(
        self,
        outdir: Path,
        tensor: pd.DataFrame,
        regime_assignments: list[tuple[int, int]],
        elapsed_times: list[float],
        encoder: OrdinalEncoder,
    ):
        """
        Save all of parameters for CubeScope
        """
        super(HeteroComp, self).save(outdir, tensor, regime_assignments, elapsed_times)
        for mode, A in enumerate(self.categoricalA):
            modedir = outdir / A.mode_idx
            modedir.mkdir(exist_ok=True)
            A.save(modedir, self.full_counterM[mode], encoder.categories_[mode])
        for mode, C in enumerate(self.continuousC):
            modedir = outdir / C.mode_idx
            modedir.mkdir(exist_ok=True)
            C.save(modedir, self.full_counterM[mode + self.nmode_cate])
        self.B.save(outdir)
        if self.anomaly:
            if len(self.chisquare_scores) > 0:
                np.savetxt(outdir / "chisquare_scores.txt", self.chisquare_scores)
            if len(self.detected_times) > 0:
                np.savetxt(outdir / "detected_times.txt", self.detected_times, fmt="%s")
        np.savetxt(outdir / "best_llh.txt", self.best_likelihoods)
        np.savetxt(outdir / "assignment.txt", self.assignment_hist, fmt="%u")
        outdir /= "regime"
        outdir.mkdir(exist_ok=True, parents=True)
        counterTK = np.concatenate(self.hist_counterTK, axis=0)
        np.savetxt(outdir / "counterTK.txt", counterTK, fmt="%u")
        with open(outdir / "full_counterM.pkl", "wb") as f:
            pickle.dump(self.full_counterM, f)
        if len(self.counterM_hist) > 0:
            with open(outdir / "counterM_hist.pkl", "wb") as f:
                pickle.dump(self.counterM_hist, f)

    def plot_online(self, out_dir: Path, tensor: pd.DataFrame, timestamp: pd.DatetimeIndex):
        self.B.plot_online(out_dir)
        # for mode in range(self.n_modes - 1):
        #     self.modes[mode].plot_online(
        #         out_dir, tensor, self.current_assignment, self.counterM[mode + 1]
        #     )

    def plot(self, out_dir: Path, encoder: OrdinalEncoder, time_labels: np.ndarray):
        """plot the result"""
        for mode, A in enumerate(self.categoricalA):
            modedir = out_dir / A.mode_idx
            modedir.mkdir(exist_ok=True)
            A.plot(modedir, self.full_counterM[mode])
            np.savetxt(modedir / "labels.txt", encoder.categories_[mode], fmt="%s")
        for mode, C in enumerate(self.continuousC):
            modedir = out_dir / C.mode_idx
            modedir.mkdir(exist_ok=True)
            C.plot(modedir, self.full_counterM[mode + self.nmode_cate])
        self.B.plot(out_dir, time_labels)
        # plot counter (time, topic)
        counterTK = np.concatenate(self.hist_counterTK, axis=0)[: time_labels.shape[0]]
        plot_time_histgram(out_dir / "counterTK.png", self.colors, counterTK, time_labels, [f"Component {k + 1}" for k in range(self.k)], self.dataConfig.freq)

    # region (private method)

    def _init_status(self, tensor: pd.DataFrame) -> tuple[np.ndarray, int, list[np.ndarray]]:
        """Initialize Counters for current tensor"""
        counterM = [np.zeros((d, self.k), dtype=int) for d in self.n_dims]
        self.counterK = np.zeros(self.k, dtype=int)
        n_events: int = len(tensor)
        assignment = np.full(n_events, -1, dtype=int)
        Asum = tensor.groupby(tensor.keys()[0]).size()
        self.counterA[Asum.index] = Asum.values
        return assignment, n_events, counterM

    def log_likelihood_init(self) -> float:
        """compute log likelihood on the initialized phase"""
        llh = 0
        llh += self.B.log_likelihood_init(self.counterM[0], self.counterA)
        for mode in range(1, self.n_modes):
            llh += self.modes[mode - 1].log_likelihood_init(self.counterM[mode], self.counterK)
        return llh

    def log_likelihood(
        self,
    ) -> float:
        """compute log likelihood"""
        llh = 0
        llh += self.B.log_likelihood(self.counterM[0], self.counterA)
        for mode in range(1, self.n_modes):
            llh += self.modes[mode - 1].log_likelihood(self.counterM[mode], self.counterK)
        return llh

    def _calc_anomaly_score(
        self,
        tensor: pd.DataFrame,
        timestamps: np.ndarray,
        assignments: np.ndarray,
        counterM: list[np.ndarray],
        counterK: np.ndarray,
        pre_timestamp: np.ndarray,
        save_score: bool = False,
    ):
        """calculate anomaly score by chi-squared static"""
        if timestamps[0] - pre_timestamp < np.timedelta64(1, "h"):
            timestamps = np.array([pre_timestamp] + list(timestamps))
        curT = self._time_interval(timestamps, pre_timestamp)
        ZERO = 1e-10
        expect_counterK = (self.normal_counterK + counterK) * curT / (self.total_interval + curT)
        mask = expect_counterK > 0
        score = np.sum(((counterK - expect_counterK) ** 2 / (expect_counterK + ZERO))[mask])
        for mode in range(self.n_modes - 1):
            expect_M = (self.normal_counterM[mode] + counterM[mode + 1]) * curT / (self.total_interval + curT)
            mask = expect_M > 0
            score += np.sum(((counterM[mode + 1] - expect_M) ** 2 / (expect_M + ZERO))[mask])
        pval = chi2.sf(score, self.DoF)
        if pval >= 0.05:
            self.total_interval += curT
            for mode in range(self.n_modes - 1):
                self.normal_counterM[mode] += counterM[mode + 1]
            self.normal_counterK += counterK
        else:
            self.logger.info(f"anomaly detected! {[0]}")
            self.detected_times.append(to_datetime(timestamps[0]))

        if save_score:
            self.chisquare_scores.append(score)

    def _time_interval(self, timestamps, pre_timestamp=None) -> float:
        intervals, _ = split_intervals(timestamps, self.dataConfig.freq)
        curT = 0
        for i, (start, end) in enumerate(intervals):
            base = timestamps[start]
            endtime = timestamps[end - 1]
            if pre_timestamp is not None and i == 0 and timestamps[0] - pre_timestamp < np.timedelta64(1, "h"):
                base = pre_timestamp + freq_to_timedelta64(self.dataConfig.freq)
            if i == len(intervals) - 1:
                endtime += freq_to_timedelta64(self.dataConfig.freq)
            interval = datetime_base(
                endtime,
                base,
                self.dataConfig.freq,
                self.dataConfig.time_scale,
            ).item()
            curT += interval
        return curT

    # endregion (private method)


# endregion(HeteroComp)


# ========================================
# region (Gibbs sampline)
# ========================================
@numba.njit()
def _gibbs_sampling_online(
    X: Float[np.ndarray, "n_events n_mode"],
    Z: Int[np.ndarray, "n_events"],
    counterM: list[Int[np.ndarray, "dim k"]],
    counterK: Int[np.ndarray, "k"],
    K: int,
    alphas: list[np.ndarray],
    B_mean: Float[np.ndarray, "time k"],
    B_variances: Float[np.ndarray, "time k"],
    L: int,
    prev_terms: list[np.ndarray],
    mode_probs: list[Float[np.ndarray, "num_bins k"]],
    n_cate_modes: int,
    n_cont_modes: int,
):
    changed_num = 0
    ZERO = 1e-10
    for e, x in enumerate(X):
        # for each non-zero event entry, assign latent component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_, d in enumerate(x):
                counterM[mode_][d, pre_topic] -= 1

        # compute topic distribution (7)
        posts = np.ones(K)
        sample = np.array([np.random.normal(B_mean[x[0], k], np.sqrt(B_variances[x[0], k]), 5).mean() for k in range(K)])
        posts *= np.exp(sample)
        for j in range(1, n_cate_modes):  # for each categorical
            mode_prob = (counterM[j][x[j]] + prev_terms[j - 1][x[j]]) / (counterK + L * alphas[j - 1][x[j]] + ZERO)
            posts *= mode_prob
        for j in range(n_cont_modes):  # for each continuous
            mode_ = j + 1 + n_cate_modes
            posts *= mode_probs[j][x[mode_], :]
        new_topic = draw_one(posts)
        if new_topic != pre_topic:
            changed_num += 1
        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_, d in enumerate(x):
            counterM[mode_][d, new_topic] += 1
    return Z, changed_num


@numba.njit()
def _gibbs_sampling(
    X: Float[np.ndarray, "n_events n_mode"],
    Z: Int[np.ndarray, "n_events"],
    counterM: list[Int[np.ndarray, "dim k"]],
    counterK: Int[np.ndarray, "k"],
    K: int,
    alphas: list[np.ndarray],
    B_mean: Float[np.ndarray, "time k"],
    B_variances: Float[np.ndarray, "time k"],
    mode_probs: list[Float[np.ndarray, "num_bins k"]],
    n_cate_modes: int,
    n_cont_modes: int,
):
    changed_num = 0
    ZERO = 1e-10
    for e, x in enumerate(X):
        # for each non-zero event entry, assign latent topic/component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_, d in enumerate(x):
                counterM[mode_][d, pre_topic] -= 1

        # compute topic distribution (7)
        posts = np.ones(K)
        # mcmc
        sample = np.array([np.random.normal(B_mean[x[0], k], np.sqrt(B_variances[x[0], k]), 5).mean() for k in range(K)])
        # sample = B_mean * 0.5 * B_variances # expectation
        # sample = B_mean # map estimation
        posts *= np.exp(sample)
        for j in range(1, n_cate_modes):
            posts *= counterM[j][x[j]] + alphas[j - 1][x[j]]
            posts /= counterK + np.sum(alphas[j - 1]) + ZERO
        for j in range(n_cont_modes):
            mode_ = j + 1 + n_cate_modes
            posts *= mode_probs[j][x[mode_], :]
        new_topic = draw_one(posts)
        if new_topic != pre_topic:
            changed_num += 1
        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_, d in enumerate(x):
            counterM[mode_][d, new_topic] += 1
    return Z, changed_num


# endregion (Gibbs sampline)
