from dataclasses import dataclass, field

# from dataclasses_json import dataclass_json


# @dataclass_json
@dataclass
class ModelConfig:
    name: str = ""
    k: int = 1
    """num of topic"""
    width: int = 0
    verbose: bool = True
    maxl: int = 5
    iter_num: int = 20
    FB: int = 16
    Regime_R: float = 3.0e-3
    Lambda = 0.1
    max_ini_r = 2
    tol_r: float = field(init=False)
    anomaly: bool = False
    alpha: float = 1.0
    beta: float = 1.0
    learning_rate: float = 0.1
    num_bins: int = 300
    """num of bins for continuous mode"""
    C_lengthscale: int = 200


# @dataclass_json
@dataclass
class DataConfig:
    name: str = ""
    time_idx: str = ""
    freq: str = "h"
    init_len: int = 0
    categorical_idxs: list[str] = field(default_factory=list)
    continuous_idxs: list[str] = field(default_factory=list)
    label_col: str | None = None
    user_idx: str | None = None
    item_idx: str | None = None
    time_scale: float = 0.1
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save_batch: bool = False
    plot_batch: bool = False
