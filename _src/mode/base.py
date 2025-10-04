import abc
import logging
import pathlib

import numpy as np
from _src.configs import ModelConfig
import pandas as pd

ZERO = 1.0e-8


# =================================
# region(Base)
# =================================
class Base(abc.ABC):
    """base class of attribute"""

    def __init__(self, k: int, param_num: int, config: ModelConfig):
        self.config: ModelConfig = config
        self.k: int = k
        self.L: int = 0
        self.logger: logging.Logger = logging.getLogger()
        self.mode_idx: str = ""

    def setL(self, L: int):
        self.L = L

    def reset(self):
        return


    def init_prev_dist(self, l: int, *args):
        """Calculate the parameters of the posterior distribution during initialization."""
        return

    def update_prev_dist(self, *args):
        return

    def init_gibbs_batch(self, l: int, tensor: pd.DataFrame, *args) -> pd.DataFrame:
        return tensor

    def post_gibbs_batch(self, l: int, *args):
        raise NotImplementedError

    def init_gibbs(self, tensor: pd.DataFrame, *args) -> pd.DataFrame:
        return tensor

    def post_gibbs(self, *args):
        """called after gibbs sampling"""
        return

    @abc.abstractmethod
    def log_likelihood_init(self, counterM: np.ndarray, *args):
        """log likelihood of the initial phase."""
        pass

    @abc.abstractmethod
    def log_likelihood(self, counterM: np.ndarray, *args):
        """log likelihood"""
        pass

    def _compute_prev_dist(self, *args) -> np.ndarray:
        """"""
        raise NotImplementedError

    def save_online(self, output_path: pathlib.Path, *args):
        return

    def save(self, outdir: pathlib.Path, *args):
        return

    def save_history(self):
        return

    def plot(self, output_path: pathlib.Path, *args):
        return

    def plot_online(self, output_path: pathlib.Path, *args):
        return
# endregion(Base)
