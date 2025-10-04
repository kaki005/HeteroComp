import logging

import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from sklearn.preprocessing import MultiLabelBinarizer
import requests
import gc
import joblib
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import geopandas as gpd
from tqdm import tqdm
import subprocess
from itertools import combinations


def data_dir()-> Path:
    return Path("_data")

def print_dataset(
    logger: logging.Logger,
    tensor: pd.DataFrame,
    categorical_idxs: list[str],
    time_idx: str,
):
    tensor = tensor[[time_idx] + categorical_idxs]
    tensor_shape = tensor.max().values + 1
    logger.info("--Input dataset--")
    logger.info(tensor.head())
    tensor_shape = (tensor.max().values + 1).astype(int)
    # n_full_cells = len(tensor.groupby(categorical_idxs + [time_idx]).count())
    logger.info("--Dataset description--")
    logger.info(f"tensor shape : {tensor_shape}")
    logger.info(f"# of records: {len(tensor)}")
    # logger.info(f"sparsity(%): {1 - n_full_cells / np.prod(tensor_shape)}")
    logger.info("------------------------")


def load_dataset(
    data_name: str, time_idx: str, continuous_idx: list[str], categorical_idx: list[str]
) -> pd.DataFrame:
    logger = logging.getLogger()
    match data_name:
        # region (WebIDS2023)
        case "WebIDS2023":
            datadir = data_dir() / "WebIDS2023"
            processed_dir = datadir / "processed.csv.gz"
            if processed_dir.exists():
                raw_df = pd.read_csv(
                    processed_dir,
                    dtype=str,
                    usecols=continuous_idx
                    + categorical_idx
                    + [time_idx, "attack_type"],
                )
            else:
                raw_df = pd.DataFrame()
                for path in sorted(datadir.rglob("*.csv")):
                    logger.info(f"{path=}")
                    df = pd.read_csv(path)
                    for col in df:
                        logger.info(f"{col=}")
                    for idx in continuous_idx:
                        df = df[df[idx].astype(float) >= 0]  # フィルタ
                    raw_df = pd.concat([raw_df, df], axis=0)
                    del df
                    gc.collect()
                raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
                raw_df = raw_df.sort_values(by=time_idx)
                raw_df.to_csv(processed_dir, compression="gzip", index=False)
            raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
            raw_df.loc[raw_df["attack_type"] == "benign", "attack_type"] = 0
        # endregion (WebIDS2023)
        # region(CUPID)
        case "CUPID":
            datadir = Path(kagglehub.dataset_download("dhoogla/cupid-2022"))
            processed_dir = datadir / "processed.csv.gz"
            if processed_dir.exists():
                raw_df = pd.read_csv(processed_dir, dtype=str)
            else:
                raw_df = pd.DataFrame()
                for path in glob.glob(f"{datadir}/*.parquet"):
                    logger.info(f"{path=}")
                    df = pd.read_parquet(path)
                    df.columns = [str(col).strip() for col in df.columns]
                    for idx in continuous_idx:
                        df = df[df[idx].astype(float) >= 0]  # フィルタ
                    df[time_idx] = pd.to_datetime(
                        df[time_idx], format="%d/%m/%Y %I:%M:%S %p"
                    )
                    raw_df = pd.concat([raw_df, df], axis=0)
                    del df
                    gc.collect()
                raw_df[time_idx] = pd.to_datetime(
                    raw_df[time_idx], format="%d/%m/%Y %I:%M:%S %p"
                )
                raw_df = raw_df.sort_values(by=time_idx)
                raw_df.to_csv(processed_dir, compression="gzip", index=False)
            raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
            raw_df["Label"] = raw_df["Label"].astype(int)
        # endregion(CUPID)
        # region(DDoS2019)
        case "DDoS2019":
            datadir = data_dir() / "CICDDoS2019"
            processed_dir = datadir / "processed.csv.gz"
            if processed_dir.exists():
                raw_df = pd.read_csv(
                    processed_dir,
                    dtype=str,
                    usecols=continuous_idx + categorical_idx + [time_idx, "Label"],
                )
                logger.info(f"{raw_df.head()}")
            else:
                raw_df = pd.DataFrame()
                for directory in ["03-11", "01-12"]:
                    for path in glob.glob(f"{datadir / directory}/*.csv"):
                        logger.info("==========================")
                        logger.info(f"{path=}")
                        df = pd.read_csv(path)
                        df.columns = [str(col).strip() for col in df.columns]
                        for idx in continuous_idx:
                            df = df[df[idx].astype(float) >= 0]  # filter
                            df = _remove_na_or_inf(df, idx, logger)
                        raw_df = pd.concat([raw_df, df], axis=0)
                        del df
                        gc.collect()
                raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
                raw_df = raw_df.sort_values(by=time_idx)
                raw_df.to_csv(processed_dir, compression="gzip", index=False)
            raw_df.loc[raw_df["Label"] == "BENIGN", "Label"] = 0
        # endregion(DDoS2019)
        # region (Edge)
        case "Edge":
            logger.info(f"{continuous_idx=}")
            datadir = Path(
                kagglehub.dataset_download(
                    "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
                )
            )
            raw_df = pd.read_csv(
                datadir
                / "Edge-IIoTset dataset"
                / "Selected dataset for ML and DL"
                / "DNN-EdgeIIoT-dataset.csv",
                engine="python"
            )
            raw_df.loc[raw_df["Attack_type"] == "Normal", "Attack_type"] = 0
            raw_df[time_idx] = pd.to_datetime(raw_df[time_idx], errors="coerce")
            raw_df = raw_df.sort_values(by=time_idx)
            raw_df = raw_df.dropna(subset=[time_idx])
            raw_df = raw_df[raw_df[time_idx].dt.year == 2021]  # 異常データをfilt
            raw_df[categorical_idx] = raw_df[categorical_idx].astype(str)
            raw_df.loc[raw_df["Attack_type"] == "Normal", "Attack_type"] = 0
        # endregion (Edge)
        # region (cidds2018)
        case "cidds2018":
            datadir = data_dir() / "CIDDS2018"
            processed_dir = datadir / "processed.csv.gz"
            if processed_dir.exists():
                raw_df = pd.read_csv(
                    processed_dir,
                    dtype=str,
                    usecols=continuous_idx + categorical_idx + [time_idx, "Label"],
                )
                logger.info(f"{raw_df.head()}")
            else:
                list_patch = np.sort(glob.glob(f"{datadir}/*.csv", recursive=True))
                with Pool(processes=cpu_count()) as pool:
                    func = partial(
                        process_cidds2018_csv,
                        time_idx=time_idx,
                        continuous_idx=continuous_idx,
                    )
                    raw_df_list = list(
                        tqdm(pool.imap(func, list_patch), total=len(list_patch))
                    )
                    raw_df = pd.concat(
                        [df for df in raw_df_list if not df.empty], axis=0
                    )
                    raw_df["Label"] = raw_df["Label"].str.replace(
                        r" - Attempted$", "", regex=True
                    )  # clean label
                    del raw_df_list
                    raw_df.to_csv(processed_dir, compression="gzip", index=False)
                for idx in continuous_idx:  # foreach continuous attribute
                    logger.info(f"{idx} : {raw_df[idx].astype(float).min()}")
            raw_df.loc[raw_df["Label"] == "BENIGN", "Label"] = 0
        # endregion (cidds2018)
        # region (cidds2017)
        case "cidds2017":
            datadir = data_dir() / "CIDDS2017"
            processed_dir = datadir / "processed.csv.gz"
            if processed_dir.exists():
                raw_df = pd.read_csv(
                    processed_dir,
                    dtype=str,
                    usecols=continuous_idx + categorical_idx + [time_idx, "Label"],
                )
            else:
                filenames = [
                    "monday.csv",
                    "tuesday.csv",
                    "wednesday.csv",
                    "thursday.csv",
                    "friday.csv",
                ]
                raw_df = pd.DataFrame()
                for file in filenames:
                    logger.info(f"load {filenames}")
                    df = pd.read_csv(datadir / file)
                    for idx in continuous_idx:  # foreach continous attribute
                        df = df[df[idx].astype(float) >= 0]  # filter
                    df = df.sort_values(by=time_idx)
                    raw_df = pd.concat([raw_df, df], axis=0)
                    raw_df["Label"] = raw_df["Label"].str.replace(
                        r" - Attempted$", "", regex=True
                    )  # clean label
                    del df
                    gc.collect()
                raw_df.to_csv(processed_dir, index=False, compression="gzip")
            for idx in continuous_idx:  # foreach continous attribute
                logger.info(f"{idx} : {raw_df[idx].astype(float).min()}")
            raw_df.loc[raw_df["Label"] == "BENIGN", "Label"] = 0
        # endregion (cidds2017)
        # region (amazon_movieTV)
        case "amazon_movieTV":
            dir_path = data_dir() / "amazon"
            preprocess_path = Path(dir_path / "Movies_and_TV_process.csv.gz")
            if preprocess_path.exists():
                raw_df = pd.read_csv(preprocess_path)
            else:
                print("load meta_df")
                meta_df = pd.read_json(
                    dir_path / "meta_Movies_and_TV.jsonl.gz",
                    lines=True,
                )
                meta_df = meta_df.drop(
                        columns=["subtitle","average_rating","rating_number","features","description","images","bought_together","videos","author", "store"]
                    )
                for idx in continuous_idx:
                    meta_df = _remove_na_or_inf(meta_df, idx, logger)
                meta_df = meta_df[meta_df["price"].astype(float) > 0]
                meta_df["Genre"] = meta_df["details"].apply(lambda d: str(d.get("Genre")).split(",") if isinstance(d, dict) and "Genre" in d else [])
                logger.info(f"create dummies")
                genre_dummies = series_to_dummies(meta_df, "categories")
                genre_dummies = _split_dummies(genre_dummies, ',')
                genre_dummies = pd.concat([genre_dummies, series_to_dummies(meta_df, "Genre")], axis=1)
                meta_df = meta_df.drop(columns=["Genre"])
                genre_dummies.columns = genre_dummies.columns.map(lambda a: str(a).strip().lower())  # trim space
                genre_dummies = genre_dummies.groupby(level=0, axis=1).max()  # mege same column name
                logger.info(F"{genre_dummies.shape=}")
                for separator in ["&", "/", '|', ';', '>', '_', "see more"]:
                    logger.info(F"{separator=}")
                    genre_dummies = _split_dummies(genre_dummies, separator)
                genre_dummies = genre_dummies.drop(columns=["", "tv", "genre for featured categories", "movies", "drama",
                                                            "60 minutes store", "all", "aerosmith", "best of 2013", "a", "e home video", "holidays", "seasonal",
                                                            "all disney titles", "by country", "entertainment", "general", "special interests", "special interest",
                                                            "featured deals","featured categories", "20th century fox home entertainment", "all bbc titles", "pbs", "boxed sets",
                                                            "all fox titles", "all hbo titles", "all mgm titles", "all titles", "eerie", "dvd", "hbo", "bbc", "special editions",
                                                            "unscripted",'independently distributed',
                                                            "all a", "today's deals", "e titles", "foreign films", "fully loaded dvds", "television", "new releases", "dts", 'dvd movie'])
                genre_dummies = merge_dummies(genre_dummies, "anime", ["animation", "animated", "animated movies","animated movie","animated series", "anime movie", "anime series", "animated characters", "animated cartoon",
                                                                       "animated cartoons", "computer animation", "scooby doo animated movies",
                                                                       "shrek", "one piece", "minions", "dreamworks animation"])
                genre_dummies = merge_dummies(genre_dummies, "science fiction", ["sci-fi","sci-fi series", "classic sci-fi", "science", "sci-fi action", "sci fi channel", "all sci fi channel shows",
                                                                                 "10-12 yearsscience fiction", "adventurescience fiction", "adventuredvd movie"])
                genre_dummies = merge_dummies(genre_dummies, "horror", ["scary", "anchor bay horror store", "thrillers", "thrilling", "thriller movie", "spy thriller", "serious", "horror movie"])
                genre_dummies = merge_dummies(genre_dummies, "arts", ["artists", "art house", "art", "performing arts", "arthouse"])
                genre_dummies = merge_dummies(genre_dummies, "adventure", ["adventures", "adventure series", "adventure movie", "adventuredrama", "adventurerent movie", "adventuredetestable moi", "space adventure", "adventuretelevision"])
                genre_dummies["adventure"] |= genre_dummies["action adventure"]
                genre_dummies = merge_dummies(genre_dummies, "documentary", ["documentaries"])
                genre_dummies = merge_dummies(genre_dummies, "fantasy", ["fantastic", "fantasy series"])
                genre_dummies = merge_dummies(genre_dummies, "romance", ["romantic", "love", "romance movie"])
                genre_dummies = merge_dummies(genre_dummies, "suspense", ["suspense movie", "suspense", "suspensedrama", "suspensehorror", "suspensescience fiction", "suspensedvd movie"])
                genre_dummies = merge_dummies(genre_dummies, "kids", ["family", "family entertainment","family features", "family friendly", "family movie", "children's","blu-ray moviekids", "classics kids love", "shakespeare for kids", "children", '10-12 years', '3-6 years', '7-9 years', '7-11 years', "10-12 yearskids"])
                genre_dummies = merge_dummies(genre_dummies, "fitness", ["sports", "exercise", "yoga", "yoga studios", "yoga journal", "yoga zone"])
                genre_dummies = merge_dummies(genre_dummies, "westerns", ["western"])
                genre_dummies = merge_dummies(genre_dummies, "military", ["war", "wars", "military and war", "korean war", "vietnam war"])
                genre_dummies = merge_dummies(genre_dummies, "romance", ["romantic comedy movie", "romantic-drama"])
                genre_dummies["romance"] |= genre_dummies["romantic comedy"]
                genre_dummies = merge_dummies(genre_dummies, "comedy", ["romantic comedy", "classic comedies", "the comedies", "comedy series", "romantic comedies", "comedia", "comedy movie", "comedy romance", "comedydrama"])
                genre_dummies = merge_dummies(genre_dummies, "religion", ["spirituality", "religious", "christian movies", "christian", "christain", "faith", "faith and spirituality", "faith-based", "christian ministry", "christian living", "christian education", "christianity",
                                                                          "christian video", "cult movies", "bible", "bible commentary", "bible study", "bible study guides"])
                genre_dummies = merge_dummies(genre_dummies, "sony pictures home entertainment", ["all sony pictures titles", "sony pictures classics"])
                genre_dummies = merge_dummies(genre_dummies, "universal studios home entertainment", ["all universal studios titles"])
                genre_dummies = merge_dummies(genre_dummies, "lionsgate home entertainment", ["all lionsgate titles"])
                genre_dummies = merge_dummies(genre_dummies, "mystery", ["mysterious", "mystical", "mystery movie", "mystery thrillers", "mystery series", "mystery…", "myster…"])
                genre_dummies = merge_dummies(genre_dummies, "classical", ["classics", "classic", "classic tv"])
                genre_dummies = merge_dummies(genre_dummies, "musicals", ["musical", "musical instruments"])
                genre_dummies = merge_dummies(genre_dummies, "history", ["historical context", "history", "history channel"])
                genre_dummies = merge_dummies(genre_dummies, "action", ["hong kong action", "live action", "action movie", "action series", "action adventure"])
                genre_dummies = merge_dummies(genre_dummies, "music", ["music artists","music videos", "music video", "guitar","world music", "concerts", "soundtracks", "movie soundtracks","hip-hop", "music videos and concerts", "opera", "performing arts movies", "children's music", "rock-music", "rock"])
                genre_dummies = genre_dummies.drop(columns=[col for col in genre_dummies.columns if 'blu-ray' in col])
                genre_dummies = _split_dummies(genre_dummies, '-')
                genre_dummies = genre_dummies.loc[:, (genre_dummies == 1).sum(axis=0) >= 1000]
                meta_df = pd.concat(
                    [meta_df.drop(columns=["categories", "details"]), genre_dummies], axis=1
                )
                logger.info("load review_df")
                raw_df = pd.DataFrame()
                df_array = pd.read_json(dir_path / "Movies_and_TV.jsonl.gz", lines=True, chunksize=1000000,)
                for df in df_array:
                    df = df.drop(
                        columns=["title","text", "images", "verified_purchase", "helpful_vote",]
                    )
                    review_df = pd.merge(df, meta_df, on="parent_asin", how="left")
                    review_df = review_df.drop(columns=[ "parent_asin"])
                    for idx in continuous_idx:
                        review_df = _remove_na_or_inf(review_df, idx, logger)
                    raw_df = pd.concat([raw_df, review_df], axis=0)
                    del df
                    del review_df
                    gc.collect()
                raw_df.to_csv(preprocess_path, compression="gzip", index=False)
                gc.collect()
            raw_df = raw_df[raw_df["rating"] >= 3.0]
            raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
            raw_df = raw_df[raw_df[time_idx].dt.year == 2020]
            import re
            raw_df["title"] = raw_df["title"].astype(str).map(lambda a: re.split(r'[\(\[]', a)[0].strip())
        # endregion (amazon_movieTV)
        case _:
            raise Exception(f"not data : {data_name}")
    raw_df[continuous_idx] = raw_df[continuous_idx].astype(float)
    raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])
    return raw_df





def _split_dummies(dummies: pd.DataFrame, separator: str) -> pd.DataFrame:
    new_cols = {}
    drop_cols = []
    for col in dummies.columns:
        if separator in col:
            parts = [s.strip() for s in col.split(separator) if s.strip()]
            for part in parts:
                if part == "":
                    continue
                new_cols[part] = new_cols.get(part, dummies.get(part, 0)) | dummies[col]
            drop_cols.append(col)
    dummies = dummies.drop(columns=drop_cols)
    for k, v in new_cols.items():
        dummies[k] = v
    return dummies





def process_cidds2018_csv(path, time_idx, continuous_idx):
    try:
        print(f"load {path}")
        df = pd.read_csv(path, dtype=str)
        for idx in continuous_idx:  # 連続モードごとに
            df = df[df[idx].astype(float) >= 0]  # フィルタ
        df = df.astype(str)
        df = df[df[time_idx].apply(pd.to_datetime).dt.year >= 2018]  # 欠損データをfilt
        df = df.sort_values(by=time_idx)
        return df.astype(str)

    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return pd.DataFrame()  # 空で返す




def dummy_to_categorical(df: pd.DataFrame, dummy_cols: list[str]) -> pd.Series:
    row_sums = df[dummy_cols].sum(axis=1)

    # 重複があるかどうかチェック
    if (row_sums > 1).any():
        details = []
        for idx, row in df.iterrows():
            active = [col for col in dummy_cols if row[col] == 1]
            if len(active) > 1:
                details.append(f"行 {idx}: {active}")

        msg = "1行に複数カテゴリが立っています:\n" + "\n".join(details)
        raise ValueError(msg)

    # 問題なければカテゴリに変換
    return df[dummy_cols].idxmax(axis=1)

def _remove_na_or_inf(
    df: pd.DataFrame, col_name: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    指定されたDataFrameの列から、NaNまたは無限大を含む行を削除します。

    Args:
        df (pd.DataFrame): 処理対象のDataFrame。
        col_name (str): NaNまたは無限大をチェックする列の名前。
        logger (logging.Logger): ロギングに使用するロガーオブジェクト。

    Returns:
        pd.DataFrame: NaNまたは無限大の行が削除された新しいDataFrame。
                      元のDataFrameは変更されません。
    """
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df = df.dropna(subset=[col_name])
    is_inf = np.isinf(df[col_name])  # 2. 無限大 (inf, -inf) のチェック
    is_na = df[col_name].isna()  # 3. 欠損値 (NaN) のチェック
    rows_to_drop = is_inf | is_na  # 4. 削除対象の行を特定
    return df[~rows_to_drop]  # 6. 該当する行を削除して新しいDataFrameを返す



def merge_dummies(df: pd.DataFrame, base_col: str, merge_cols: list[str]):
    for col in merge_cols:
        if col not in df.columns:
            continue
        if base_col in df.columns:
            df[base_col] = df[base_col] | df[col]
        else:
            df[base_col] = df[col]
    return df.drop(columns=merge_cols)
