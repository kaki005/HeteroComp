import pandas as pd
from _src.configs import Config
from . import Basemodel, HeteroComp, CubeScope, CyberCScope
import numpy as np
from .CubeScope import CubeScope
from .CyberCScope import CyberCScope
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
        case "cubescope":
            return CubeScope(
                tensor,
                config.model.k,
                config.model.width,
                config.data.init_len,
                config.model.verbose,
                categorical_idxs,
                config.model.alpha,
                config.model.beta,
                config.model.anomaly,
                tensor_shape=tensor_shape,
            )
        case "cybercscope":
            return CyberCScope(
                tensor,
                config.model.k,
                config.model.width,
                config.data.init_len,
                config.data.time_idx,
                list(config.data.categorical_idxs),
                list(config.data.continuous_idxs),
                config.model,
                early_stoppping = True,
            )
        case "heterocomp":
            return HeteroComp(tensor, config.model, config.data, mode_bounds)
        case _:
            raise Exception()
