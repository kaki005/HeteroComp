import numpy as np
import pandas as pd

from _src.configs import Config

from .basemodel import Basemodel
from .HeteroComp import HeteroComp


def load_model(
    tensor: pd.DataFrame,
    config: Config,
    model_name: str,
    tensor_shape: np.ndarray,
    anom_series: pd.DataFrame,
    categorical_idxs: list[str],
    mode_bounds: list[np.ndarray],
) -> Basemodel:
    match model_name:
        case "heterocomp":
            return HeteroComp(tensor, config.model, config.data, mode_bounds)
        case _:
            raise Exception()
